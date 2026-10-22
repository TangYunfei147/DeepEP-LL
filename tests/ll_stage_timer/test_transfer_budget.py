"""The ten-send allowance applies to the suite, across repeated exports."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('ll_transfer_budget', Path(__file__).with_name('transfer_budget.py'))
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def issue(suite, kind, values, output=None):
    with budget.transaction(suite, kind, [digest(value) for value in values]) as result:
        if output:
            output.mkdir(parents=True)
            for value in values:
                (output / f'{value}.txt').write_text(str(value))
    return result


def test_initial_state_is_read_only_and_reserves_summary(tmp_path):
    suite = tmp_path / 'suite'
    assert budget.budget_state(suite) == dict(limit=10, issued=0, reserved=1, remaining=9)
    assert not suite.exists()


def test_details_before_summary_cannot_consume_its_slot(tmp_path):
    issue(tmp_path, 'detail', range(9))
    assert budget.budget_state(tmp_path) == dict(limit=10, issued=9, reserved=1, remaining=0)
    with pytest.raises(budget.TransferBudgetError, match='No files were issued'):
        issue(tmp_path, 'detail', [9])
    result = issue(tmp_path, 'summary', ['summary'])
    assert result['new_count'] == 1 and result['remaining'] == 0
    assert budget.budget_state(tmp_path) == dict(limit=10, issued=10, reserved=0, remaining=0)


def test_cumulative_limit_across_calls_and_output_directories(tmp_path):
    suite = tmp_path / 'suite'
    issue(suite, 'summary', ['summary'])
    for index in range(3):
        issue(suite, 'detail', range(index * 3, index * 3 + 3), tmp_path / f'out{index}')
    refused = tmp_path / 'out-refused'
    with pytest.raises(budget.TransferBudgetError, match='limit is 10'):
        issue(suite, 'detail', [100], refused)
    assert not refused.exists()
    assert budget.budget_state(suite)['issued'] == 10
    assert not (suite / 'triage/send_budget.lock').exists()


def test_identical_payload_in_new_output_directory_is_not_charged_twice(tmp_path):
    suite = tmp_path / 'suite'
    issue(suite, 'detail', [1, 2], tmp_path / 'first')
    result = issue(suite, 'detail', [1, 2, 2], tmp_path / 'second')
    assert result['new_count'] == 0
    assert result['already_issued'] == [digest(1), digest(2)]
    assert result['artifact_count'] == 2
    assert budget.budget_state(suite)['issued'] == 2


def test_summary_revisions_are_new_sends_but_identical_summary_is_free(tmp_path):
    result = issue(tmp_path, 'summary', ['summary1'])
    assert result['remaining'] == 9
    result = issue(tmp_path, 'summary', ['summary1'])
    assert result['new_count'] == 0 and result['remaining'] == 9
    result = issue(tmp_path, 'summary', ['summary2'])
    assert result['new_count'] == 1 and result['remaining'] == 8
    issue(tmp_path, 'detail', range(8))
    with pytest.raises(budget.TransferBudgetError):
        issue(tmp_path, 'summary', ['summary3'])


def test_identity_includes_artifact_kind(tmp_path):
    issue(tmp_path, 'summary', ['same'])
    issue(tmp_path, 'detail', ['same'])
    assert budget.budget_state(tmp_path)['issued'] == 2


def test_oversize_batch_rejected_before_writing_any_files(tmp_path):
    output = tmp_path / 'out'
    with pytest.raises(budget.TransferBudgetError):
        issue(tmp_path / 'suite', 'detail', range(10), output)
    assert not output.exists()
    assert not list((tmp_path / 'suite/triage').iterdir())


@pytest.mark.parametrize('document', [
    'not json', '{}', '[]',
    '{"version":true,"limit":10,"artifacts":[]}',
    '{"version":1,"limit":11,"artifacts":[]}',
    '{"version":1,"limit":10,"artifacts":{}}',
    '{"version":1,"limit":10,"artifacts":""}',
    '{"version":1,"limit":10,"artifacts":null}',
    '{"version":1,"limit":10,"artifacts":{},"extra":0}',
    json.dumps(dict(version=1, limit=10, artifacts=[dict(kind='detail', sha256='bad')])),
    json.dumps(dict(version=1, limit=10, artifacts=[dict(kind=[], sha256=digest(1))])),
    json.dumps(dict(version=1, limit=10, artifacts=[dict(kind='detail', sha256=digest(1))] * 2)),
    json.dumps(dict(version=1, limit=10, artifacts=[dict(kind='detail', sha256=digest(i)) for i in range(10)])),
])
def test_invalid_ledger_fails_closed_without_reset(tmp_path, document):
    directory = tmp_path / 'triage'
    directory.mkdir()
    ledger = directory / 'send_budget.json'
    ledger.write_text(document)
    with pytest.raises(budget.TransferBudgetError, match='Invalid send budget'):
        budget.budget_state(tmp_path)
    with pytest.raises(budget.TransferBudgetError, match='Invalid send budget'):
        issue(tmp_path, 'summary', ['x'])
    assert ledger.read_text() == document
    assert not (directory / 'send_budget.lock').exists()


def test_concurrent_export_is_rejected_and_original_lock_retained(tmp_path):
    with budget.transaction(tmp_path, 'summary', [digest('summary')]):
        with pytest.raises(budget.TransferBudgetError, match='locked'):
            issue(tmp_path, 'detail', [1])
        assert (tmp_path / 'triage/send_budget.lock').exists()
    assert not (tmp_path / 'triage/send_budget.lock').exists()
    assert budget.budget_state(tmp_path)['issued'] == 1


def test_stale_lock_is_never_automatically_deleted(tmp_path):
    directory = tmp_path / 'triage'
    directory.mkdir()
    lock = directory / 'send_budget.lock'
    lock.write_text('an interrupted export')
    with pytest.raises(budget.TransferBudgetError, match='locked'):
        issue(tmp_path, 'summary', ['x'])
    assert lock.read_text() == 'an interrupted export'


def test_writer_failure_does_not_charge_budget(tmp_path):
    issue(tmp_path, 'summary', ['summary'])
    with pytest.raises(OSError, match='output failure'):
        with budget.transaction(tmp_path, 'detail', [digest(1)]):
            raise OSError('output failure')
    assert budget.budget_state(tmp_path)['issued'] == 1
    assert not (tmp_path / 'triage/send_budget.lock').exists()


def test_atomic_commit_failure_keeps_previous_ledger(tmp_path, monkeypatch):
    issue(tmp_path, 'summary', ['summary'])
    ledger = tmp_path / 'triage/send_budget.json'
    before = ledger.read_bytes()
    def fail_replace(*args):
        raise OSError('disk full')
    monkeypatch.setattr(budget.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='disk full'):
        issue(tmp_path, 'detail', [1])
    assert ledger.read_bytes() == before
    assert budget.budget_state(tmp_path)['issued'] == 1
    assert list((tmp_path / 'triage').iterdir()) == [ledger]


@pytest.mark.parametrize('kind,values', [('other', [digest(1)]), ('detail', []),
    ('detail', 'a' * 64), ('detail', ['a']), ('detail', [1]),
    ('detail', ['A' * 64]), ('summary', [digest(1), digest(2)])])
def test_invalid_request_has_no_filesystem_effects(tmp_path, kind, values):
    suite = tmp_path / 'absent'
    with pytest.raises(budget.TransferBudgetError):
        with budget.transaction(suite, kind, values):
            pytest.fail('invalid request entered transaction')
    assert not suite.exists()


def test_legacy_root_summary_is_adopted_only_inside_transaction(tmp_path):
    previous = b'Legacy summary already issued\r\n'
    (tmp_path / 'SEND_FIRST.txt').write_bytes(previous)
    # A status query must remain read-only and cannot commit a migration.
    assert budget.budget_state(tmp_path) == dict(limit=10, issued=0, reserved=1, remaining=9)
    assert not (tmp_path / 'triage').exists()
    result = issue(tmp_path, 'detail', [1])
    assert result['issued_before'] == 1 and result['issued'] == 2
    assert result['new_count'] == 1 and result['reserved'] == 0 and result['remaining'] == 8
    ledger = json.loads((tmp_path / 'triage/send_budget.json').read_text())
    assert dict(kind='summary', sha256=hashlib.sha256(previous).hexdigest()) in ledger['artifacts']
    assert len(ledger['artifacts']) == 2
    assert (tmp_path / 'SEND_FIRST.txt').read_bytes() == previous


def test_migration_commits_even_when_requested_summary_is_already_issued(tmp_path):
    previous = b'Same legacy summary\n'
    (tmp_path / 'SEND_FIRST.txt').write_bytes(previous)
    previous_digest = hashlib.sha256(previous).hexdigest()
    with budget.transaction(tmp_path, 'summary', [previous_digest]) as result:
        assert result['new_count'] == 0
        assert result['already_issued'] == [previous_digest]
        assert result['issued'] == 1
    assert budget.budget_state(tmp_path) == dict(limit=10, issued=1, reserved=0, remaining=9)
    assert (tmp_path / 'triage/send_budget.json').is_file()


def test_unrecorded_root_revision_is_charged_on_repeated_detail_export(tmp_path):
    issue(tmp_path, 'summary', ['previous summary'])
    issue(tmp_path, 'detail', [1])
    (tmp_path / 'SEND_FIRST.txt').write_text('unrecorded new summary')
    assert budget.budget_state(tmp_path)['issued'] == 2
    result = issue(tmp_path, 'detail', [1])
    assert result['new_count'] == 0
    assert result['issued_before'] == result['issued'] == 3
    assert budget.budget_state(tmp_path)['issued'] == 3


def test_adopted_legacy_summary_does_not_commit_when_output_fails(tmp_path):
    previous = b'Legacy summary'
    (tmp_path / 'SEND_FIRST.txt').write_bytes(previous)
    with pytest.raises(OSError, match='output failed'):
        with budget.transaction(tmp_path, 'detail', [digest(1)]):
            raise OSError('output failed')
    assert not (tmp_path / 'triage/send_budget.json').exists()
    assert (tmp_path / 'SEND_FIRST.txt').read_bytes() == previous
    assert issue(tmp_path, 'detail', [1])['issued'] == 2


def test_unrecorded_root_revision_cannot_bypass_exhausted_budget(tmp_path):
    issue(tmp_path, 'summary', ['previous summary'])
    issue(tmp_path, 'detail', range(9))
    ledger = tmp_path / 'triage/send_budget.json'
    before = ledger.read_bytes()
    (tmp_path / 'SEND_FIRST.txt').write_text('unrecorded new summary')
    with pytest.raises(budget.TransferBudgetError):
        with budget.transaction(tmp_path, 'detail', [digest(1)]):
            pytest.fail('Migration beyond the suite allowance must not enter output code')
    assert ledger.read_bytes() == before
    assert not (tmp_path / 'triage/send_budget.lock').exists()


def test_failure_to_read_legacy_summary_cannot_skip_its_charge(tmp_path, monkeypatch):
    previous = tmp_path / 'SEND_FIRST.txt'
    previous.write_text('legacy summary')
    original = Path.read_bytes
    def fail_summary_read(path):
        if path == previous:
            raise OSError('legacy summary unreadable')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', fail_summary_read)
    with pytest.raises((OSError, budget.TransferBudgetError), match='unreadable'):
        with budget.transaction(tmp_path, 'detail', [digest(1)]):
            pytest.fail('Unreadable old summary must prevent output')
    assert not (tmp_path / 'triage/send_budget.json').exists()
    assert not (tmp_path / 'triage/send_budget.lock').exists()
