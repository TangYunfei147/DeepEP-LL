"""Persistent per-suite limit for artifacts issued for manual transmission.

The producer cannot observe actual pastes.  It therefore counts issued artifacts,
reserves the first summary slot, and treats every new summary or detail digest as
another potential send.  Copying an identical artifact into another output
directory never creates a second charge.  The ledger is local bookkeeping and
must not itself be included in the send list.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


LIMIT = 10
VERSION = 1
_KINDS = frozenset(('summary', 'detail'))
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


class TransferBudgetError(ValueError):
    """No artifacts may be issued until the budget error is resolved."""


def _paths(suite):
    directory = Path(suite) / 'triage'
    return directory, directory / 'send_budget.json', directory / 'send_budget.lock'


def _state(artifacts):
    issued = len(artifacts)
    reserved = 0 if any(item['kind'] == 'summary' for item in artifacts) else 1
    return dict(limit=LIMIT, issued=issued, reserved=reserved,
                remaining=LIMIT - issued - reserved)


def _load(path):
    if not path.exists():
        return dict(version=VERSION, limit=LIMIT, artifacts=[])
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(document, dict) or set(document) != {'version', 'limit', 'artifacts'}:
            raise ValueError('unexpected fields')
        if (type(document['version']) is not int or document['version'] != VERSION
                or type(document['limit']) is not int or document['limit'] != LIMIT):
            raise ValueError('version or limit mismatch')
        artifacts = document['artifacts']
        if not isinstance(artifacts, list):
            raise ValueError('artifacts must be a list')
        seen = set()
        for item in artifacts:
            if not isinstance(item, dict) or set(item) != {'kind', 'sha256'}:
                raise ValueError('invalid artifact fields')
            if item['kind'] not in _KINDS or not isinstance(item['sha256'], str) or not _DIGEST.fullmatch(item['sha256']):
                raise ValueError('invalid artifact identity')
            identity = item['kind'], item['sha256']
            if identity in seen:
                raise ValueError('duplicate artifact identity')
            seen.add(identity)
        if _state(artifacts)['remaining'] < 0:
            raise ValueError('budget already exceeded')
        return document
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise TransferBudgetError(
            'Invalid send budget ledger; refusing to issue files or reset the budget: '
            + str(error)) from error


def budget_state(suite):
    """Read committed usage without creating directories or modifying files.

    ``remaining`` excludes the reserved first summary, so a new suite starts
    with nine available detail slots.  A writer in progress is not yet counted.
    """
    _, path, _ = _paths(suite)
    return _state(_load(path)['artifacts'])


def _write_ledger(path, document):
    temporary = path.with_name('.send_budget-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as output:
            json.dump(document, output, ensure_ascii=True, sort_keys=True, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def transaction(suite, kind, artifact_digests):
    """Reserve a batch, hold an exclusive lock during output, then commit it.

    Use lowercase SHA256 identities for the exact transferable artifacts.  The
    context yields projected usage plus ``new_count``, ``new_digests`` and
    ``already_issued`` lists.  Repeated identities within the batch are deduped.
    Callers should explicitly tell users not to resend ``already_issued`` files.
    A summary batch contains exactly one digest.  The first summary consumes the
    initially reserved slot; later changed summaries consume additional slots.

    Validation, quota and lock errors occur before the body is entered.  If the
    body fails, no quota is committed.  If ledger commit fails after output, the
    caller must not advertise the output as successfully issued.  Stale locks
    and corrupt ledgers fail closed; this module never resets either one.
    """
    if kind not in _KINDS:
        raise TransferBudgetError("Artifact kind must be 'summary' or 'detail'")
    if isinstance(artifact_digests, (str, bytes)):
        raise TransferBudgetError('Artifact digests must be a nonempty sequence of SHA256 strings')
    try:
        digests = list(artifact_digests)
    except TypeError as error:
        raise TransferBudgetError('Artifact digests must be a nonempty sequence of SHA256 strings') from error
    if not digests or any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in digests):
        raise TransferBudgetError('Artifact digests must be nonempty lowercase SHA256 strings')
    digests = list(dict.fromkeys(digests))
    if kind == 'summary' and len(digests) != 1:
        raise TransferBudgetError('Issue exactly one summary artifact per transaction')

    directory, ledger_path, lock_path = _paths(suite)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise TransferBudgetError(
            'Send budget is locked by another export; no files were issued. '
            'Inspect an interrupted export before handling its stale lock.') from error
    os.close(lock_descriptor)
    try:
        document = _load(ledger_path)
        artifacts = list(document['artifacts'])
        # Conservatively count a report issued by an older producer.  Do this
        # under the same lock, so upgrading or choosing a new output directory
        # cannot silently replenish the suite's allowance.
        previous_report = Path(suite) / 'SEND_FIRST.txt'
        if previous_report.is_file():
            previous = dict(kind='summary', sha256=hashlib.sha256(previous_report.read_bytes()).hexdigest())
            if previous not in artifacts:
                artifacts.append(previous)
        before = _state(artifacts)
        known = {(item['kind'], item['sha256']) for item in artifacts}
        new_digests = [value for value in digests if (kind, value) not in known]
        already_issued = [value for value in digests if (kind, value) in known]
        proposed = artifacts + [dict(kind=kind, sha256=value) for value in new_digests]
        after = _state(proposed)
        if after['remaining'] < 0:
            available = before['remaining'] + (before['reserved'] if kind == 'summary' else 0)
            raise TransferBudgetError(
                f'Suite transmission limit is {LIMIT}: {before["issued"]} artifacts already issued, '
                f'{before["reserved"]} summary slot reserved, {available} available for this batch; '
                f'{len(new_digests)} new artifacts requested. No files were issued.')
        result = dict(after, new_count=len(new_digests), new_digests=new_digests,
                      already_issued=already_issued, artifact_count=len(digests),
                      issued_before=before['issued'], remaining_before=before['remaining'])
        yield result
        if proposed != document['artifacts']:
            _write_ledger(ledger_path, dict(version=VERSION, limit=LIMIT, artifacts=proposed))
    finally:
        lock_path.unlink()
