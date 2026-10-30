"""Shared deterministic reference-point insertion for generated cb59 sources.

Only diagnostic blocks and their pointer plumbing are added. The normalization
removes explicitly delimited additions, not arbitrary scopes or business code.
"""
import re

BEGIN = '/* LL_REFERENCE_BEGIN */'
END = '/* LL_REFERENCE_END */'
_TOKENS = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}()]', re.S)


def diagnostic(code):
    return '\n' + BEGIN + '\n' + code + '\n' + END + '\n'


def mark(word):
    return f'if (profile_) profile_->Mark<{word}>();'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Reference insertion expected one anchor ({text.count(before)}): {before[:100]}')
    return text.replace(before, after, 1)


def span(text, name):
    matches = []
    for candidate in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        depth = 0
        for token in _TOKENS.finditer(text, text.index('(', candidate.start())):
            if token.group() == '(': depth += 1
            elif token.group() == ')':
                depth -= 1
                if not depth:
                    tail = re.match(r'\s*(?:const\s*)?\{', text[token.end():])
                    if tail: matches.append(token.end() + tail.end() - 1)
                    break
    if len(matches) != 1:
        raise ValueError(f'Expected one function body {name}: {len(matches)}')
    start = matches[0]
    depth = 0
    for token in _TOKENS.finditer(text, start):
        value = token.group()
        if value == '{': depth += 1
        elif value == '}':
            depth -= 1
            if not depth:
                return start + 1, token.start()
    raise ValueError(f'Unbalanced function: {name}')


def edit_body(text, name, edit):
    start, end = span(text, name)
    return text[:start] + edit(text[start:end]) + text[end:]


def boundary(text, name, begin, end, tail=''):
    """Mark calls including explicit early returns, preserving single-if scope."""
    finish = (mark(end) + '\n' if end is not None else '') + tail
    def edit(body):
        # Every targeted function has ordinary return statements and no lambda.
        if re.search(r'\[[^\]]*\]\s*\(', body):
            raise ValueError('Do not rewrite lambda returns')
        mask = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                      lambda m: ''.join('\n' if c == '\n' else ' ' for c in m.group()), body, flags=re.S)
        for match in reversed(list(re.finditer(r'\breturn\b[^;]*;', mask))):
            start, end = match.span()
            body = body[:start] + diagnostic('{') + diagnostic(finish) + body[start:end] + diagnostic('}') + body[end:]
        return diagnostic(mark(begin) if begin is not None else '') + body + diagnostic(finish)
    return edit_body(text, name, edit)


def add_profile(text, include, extra_public='', extra_private=''):
    text = '#include "' + include + '"\n' + text
    text = once(text, 'public:', 'public:' + diagnostic(
        '__aicore__ inline void SetProfile(LlReferenceProfile::DeviceProfile* profile) { profile_ = profile; }\n' + extra_public))
    at = text.index('private:') if 'private:' in text else text.index('    const __gm__ DispatchDedup::CommArgs* args_')
    return text[:at] + diagnostic('private:\n    LlReferenceProfile::DeviceProfile* profile_ = nullptr;\n' + extra_private + '\npublic:') + text[at:]


def strip_reference(text):
    return re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', text, flags=re.S)


def instrument_counts(text):
    text = add_profile(text, '../../common/ll_reference_profile_device.h')
    for name, begin, end in [('SendCnt',6,7), ('RecvCnt',10,11), ('CalcCumsum',12,13),
                             ('CompleteCumsum',14,15), ('Finish',20,21), ('WaitDispatch',24,25),
                             ('GetCumSum',26,27), ('WaitCumSumFlag',28,29)]:
        text = boundary(text, name, begin, end)
    return text


def instrument_kernel(text):
    text = add_profile(text, '../common/ll_reference_profile_device.h',
        '__aicore__ inline void ExportProfile() { receiver_.ExportProfile(); }')
    text = edit_body(text, 'Init', lambda body: diagnostic(
        'counts_.SetProfile(profile_); sender_.SetProfile(profile_); receiver_.SetProfile(profile_);') + body)
    text = once(text, '        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);',
                diagnostic(mark(2)) + '        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);' + diagnostic(mark(3)))
    text = once(text, '                     magic, pipe);', '                     magic, pipe);' + diagnostic(mark(4)))
    text = edit_body(text, 'Init', lambda body: body + diagnostic(mark(5)))
    text = once(text, '            PipeBarrier<PIPE_ALL>();', diagnostic(mark(16)) + '            PipeBarrier<PIPE_ALL>();' + diagnostic(mark(17)))
    return text


def instrument_sender(text):
    text = add_profile(text, '../../common/ll_reference_profile_device.h')
    text = once(text, '        if (enableDedup_)\n            SendWithDedup(input);', diagnostic(mark(38)) + '        if (enableDedup_)\n            SendWithDedup(input);')
    text = once(text, '            SendWithoutDedup(input);', '            SendWithoutDedup(input);' + diagnostic(mark(39)))
    text = edit_body(text, 'Process', lambda body: body + diagnostic(mark(40)))
    text = once(text, '        const uint32_t tokenId = token, destination = expert / localExperts_;',
        '        const uint32_t tokenId = token, destination = expert / localExperts_;' + diagnostic(
            'if (profile_) profile_->SendBegin(destination != uint32_t(args_->rank), token, firstK, expert, destination);'))
    text = once(text, '        FillTail(payload, token, firstK, mask);', '        FillTail(payload, token, firstK, mask);' + diagnostic('if (profile_) profile_->PacketPrepared();'))
    text = once(text, '        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));',
        diagnostic('if (profile_) profile_->PutBegin();') + '        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));' + diagnostic('if (profile_) profile_->PutReturned();'))
    text = instrument_member_q(text)
    return boundary(text, 'Process', 8, 9, 'if (profile_) profile_->SendTotals(lanes_);')


def instrument_member_q(text):
    needle = '                const uint32_t q = CalTokenSendExpertCnt(expert, routeIndex);'
    return edit_body(text, 'FillTail', lambda body: once(body, needle, diagnostic('''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                const bool profileSecond = profile_ && profile_->SecondPeer();
                const uint64_t profileQBegin = profileSecond ? profile_->Clock() : 0U;
#endif''') + needle + diagnostic('''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                if (profileSecond) profile_->MemberQ(profile_->Clock() - profileQBegin);
#endif''')))


def instrument_receiver(text, *, current=False, stream=False, pair=False):
    include = '../common/ll_reference_receiver_log.h' if current or stream else '../../common/ll_reference_receiver_log.h'
    text = add_profile(text, include, '''__aicore__ inline void ExportProfile() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!profile_ || !profile_->Enabled() || !profileReceiving_) return;
        trace_.Export(profile_, tasks_.Count(), tasks_.Begin(), bs_, localExperts_, lanes_);
#endif
    }''', '''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    LlReferenceProfile::ReceiverLog trace_;
    bool profileReceiving_ = false;
#endif
''')
    prepare = 'Prepare' if current else 'Process'
    text = edit_body(text, prepare, lambda body: diagnostic(mark(18) + '''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        profileReceiving_ = true;
#endif''') + body)
    # Prefix timestamps remain at the actual load/shared-view point. This is
    # intentionally different from metadata-core schema5's before-prefix scan.
    for method in ('BuildPair', 'BuildStream', 'Build'):
        pattern = re.compile(r'tasks_\.' + method + r'\([^;]+;')
        text = pattern.sub(lambda m: diagnostic(mark(30)) + m.group() + diagnostic(mark(31)), text)
    zero_tail = mark(32) + '\nif (profile_) profile_->ReceiveTotals(0U);\n' + mark(19)
    # Match existing early zero-return branches; no new business branch.
    text = re.sub(r'(if \((?:tasks_\.Count\(\)|stream->Count\(\)|pair->Count\(\)) == 0U\) \{)\s*return;',
                  lambda m: m.group(1) + diagnostic(zero_tail) + 'return;', text)
    allocator = 'Allocate' if stream else 'pipe_->InitBuffer'
    needle = f'        {allocator}(lanesBuf_, lanes_ * laneStride_);'
    budget = ('Cb59Replay::' if stream else '') + 'kEpServerDedupCountUbBytes'
    spare = f'{budget} - fixed - lanes_ * laneStride_'
    if current:
        spare = f'overlap_.enabled ? uint32_t({budget} - overlap_.totalBytes) : ({spare})'
    text = once(text, needle, diagnostic(f'''uint32_t referenceBytes = 0U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_ && profile_->Enabled()) referenceBytes = trace_.Bytes(profile_, {spare}, tasks_.Count());
#endif''') + f'        {allocator}(lanesBuf_, lanes_ * laneStride_' + diagnostic('+ referenceBytes') + ');' + diagnostic('''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (referenceBytes) trace_.Init(lanesBuf_.Get<uint32_t>()[lanes_ * laneStride_ / 4U]);
#endif'''))
    # After all existing lane setup, but before polling.
    if current:
        text = edit_body(text, 'Prepare', lambda body: body + diagnostic(mark(32)))
        text = once(text, '        if (!tasks_.PendingCount()) return false;',
            '        if (!tasks_.PendingCount()) return false;' + diagnostic('if (profile_) profile_->PollBegin();'))
        needle = 'if (probe.readyMask==0U) {  scan_+=count; return false; }'
        # Blank diagnostic removal may preserve a different number of spaces.
        pattern = r'if \(probe\.readyMask==0U\) \{\s*scan_\+=count; return false; \}'
        text, count = re.subn(pattern, lambda m: m.group() + diagnostic('''if (profile_) profile_->BatchReady();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            trace_.Ready(snapshot, probe.readyMask, count);
#endif'''), text)
        if count != 1: raise ValueError('current receive ready anchor')
        text = edit_body(text, 'TryBatch', lambda body: once(body, '        return true;', diagnostic('if (profile_) profile_->BatchDone();') + '        return true;'))
        text = edit_body(text, 'Run', lambda body: body + diagnostic(mark(19) + '\nif (profile_) profile_->ReceiveTotals(tasks_.Count());'))
    else:
        text = once(text, '        uint32_t scan = 0U, issued = 0U, consumed = 0U;', diagnostic(mark(32)) + '        uint32_t scan = 0U, issued = 0U, consumed = 0U;')
        text = once(text, '        while (tasks_.PendingCount() != 0U) {', '        while (tasks_.PendingCount() != 0U) {' + diagnostic('if (profile_) profile_->PollBegin();'))
        text = once(text, '            } while (probe.readyMask == 0U);', '            } while (probe.readyMask == 0U);' + diagnostic('''if (profile_) profile_->BatchReady();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            trace_.Ready(snapshot, probe.readyMask, count);
#endif'''))
        text = once(text, '                ++consumed;\n            }', '                ++consumed;\n            }' + diagnostic('if (profile_) profile_->BatchDone();'))
        text = edit_body(text, 'Process', lambda body: body + diagnostic('if (profile_) profile_->ReceiveTotals(tasks_.Count());\n' + mark(19)))
    text = once(text, '        ClearFlags();', '        ClearFlags();' + diagnostic(mark(35)))
    text = once(text, '        PipeBarrier<PIPE_ALL>();', '        PipeBarrier<PIPE_ALL>();' + diagnostic(mark(36)))
    header = '        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);'
    text = once(text, header, header + diagnostic('''
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_ && profile_->Enabled()) trace_.Processed(j, wire, headerOffset,
            uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset),
            tasks_.Position(j));
#endif'''))
    return text


def instrument_cb59(relative, text):
    if relative == 'dispatch_server_dedup_kernel.h': return instrument_kernel(text)
    if relative.endswith('counts.h'): return instrument_counts(text)
    if relative.endswith('_send.h'): return instrument_sender(text)
    if relative.endswith('_recv.h'): return instrument_receiver(text)
    return text
