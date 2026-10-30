"""Exact retired-module names for wheel staging cleanup and package validation."""
from pathlib import PurePosixPath
import re

RETIRED_LL_UTILITIES = (
    'll_expert_timer', 'll_stage_timer', 'll_stage_timer_bandwidth',
    'll_stage_timer_legacy', 'll_stage_timer_mte3', 'll_stage_timer_sparse',
    'll_stage_timer_systemcycle',
)


def retired_utility_filename(name, *, cache=False):
    if cache:
        return any(re.fullmatch(re.escape(stem)+r'\.[A-Za-z0-9_.-]+\.py[co]', name)
                   for stem in RETIRED_LL_UTILITIES)
    return name in {stem+suffix for stem in RETIRED_LL_UTILITIES for suffix in ('.py','.pyc','.pyo')}


def retired_wheel_entries(names):
    result=[]
    for name in names:
        parts=PurePosixPath(name).parts
        if parts[:2]!=('deep_ep','utils'):
            continue
        if len(parts)==3 and retired_utility_filename(parts[2]):
            result.append(name)
        elif len(parts)==4 and parts[2]=='__pycache__' and retired_utility_filename(parts[3],cache=True):
            result.append(name)
    return result
