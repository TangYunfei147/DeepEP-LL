"""Independent full-token and ordered-dependency checks against e4d114c."""
import importlib.util
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


projection=load('reference_business_projection','scripts/prepare_ll_no_timer.py')
reference=load('reference_business_insertion','scripts/ll_reference_instrumentation.py')


def code(text):
    return reference.strip_reference(text)


def tokens(text):
    return projection.tokens(re.sub(r'^#include[^\n]*\n','',code(text),flags=re.M))


def assert_unchanged(relative):
    before=subprocess.check_output(['git','-C',str(ROOT),'show','e4d114c:'+relative]).decode('utf-8')
    if relative in projection.ALGORITHMS:
        before=projection.transform_cpp(relative,before)
    after=(ROOT/relative).read_text(encoding='utf-8')
    assert tokens(before)==tokens(after),relative
    assert projection.synchronization(before)==projection.synchronization(code(after)),relative
    def includes(text):
        return [line for line in text.splitlines() if line.startswith('#include')
                and not any(name in line for name in ('ll_core_config.h','ll_reference_profile_device.h','ll_reference_receiver_log.h'))]
    assert includes(before)==includes(after),relative


def compact(text):
    return re.sub(r'\s+','',code(text))
