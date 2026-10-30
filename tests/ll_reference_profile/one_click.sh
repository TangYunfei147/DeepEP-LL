#!/usr/bin/env bash
# Fresh frozen checkout + full reference wheel, then dispatch capture and export.
# Execute with bash; failures return from the function, preserving shell/logs.
reference_main() {
    local root group source_dir python_bin run_name address port shmem_port action stamp
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P) || return
    if (( $# < 4 )); then
        printf '%s\n' 'Usage: bash tests/ll_reference_profile/one_click.sh RUN MASTER_ADDR PORT SHMEM_PORT [--build-only|--skip-build] [ST options...]'
        return 2
    fi
    run_name=$1; address=$2; port=$3; shmem_port=$4; shift 4
    if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ || $run_name == . || $run_name == .. ]]; then
        printf '%s\n' 'Use a new simple run name.' >&2; return 2
    fi
    action=build-run
    if [[ ${1:-} == --build-only || ${1:-} == --skip-build ]]; then action=$1; shift; fi
    group="$root/diagnostic-runs/$run_name"; source_dir="$group/source"
    if [[ -L "$root/diagnostic-runs" || -L "$group" || -L "$source_dir" || -L "$group/venv" ]]; then
        printf '%s\n' 'Run paths may not redirect outside the project.' >&2; return 2
    fi
    if [[ $action != --skip-build ]]; then
        if [[ -e $group ]]; then printf '%s\n' 'Build group exists; use --skip-build or a new run name.' >&2; return 2; fi
        local revision changes
        revision=$(git -C "$root" rev-parse HEAD) || return
        changes=$(git -C "$root" status --porcelain --untracked-files=no --ignore-submodules=dirty) || return
        if [[ -n $changes ]]; then
            printf '%s\n' 'Commit/stash intentional tracked edits before freezing the build.' >&2; return 2
        fi
        mkdir -p -- "$root/diagnostic-runs" || return
        mkdir -- "$group" || return
        mkdir -- "$group/logs" "$group/wheels" || return
        git -c core.autocrlf=false clone --no-hardlinks --dissociate --no-checkout --no-recurse-submodules "$root" "$source_dir" >"$group/logs/clone.log" 2>&1 || return
        git -C "$source_dir" -c core.autocrlf=false -c submodule.recurse=false checkout --detach "$revision" >"$group/logs/checkout.log" 2>&1 || return
        git -C "$source_dir" submodule update --init --recursive --depth 1 -- third-party/shmem >"$group/logs/shmem-source.log" 2>&1 || return
        python -m venv --system-site-packages "$group/venv" >"$group/logs/venv.log" 2>&1 || return
    fi
    python_bin="$group/venv/bin/python"
    if [[ ! -x $python_bin || ! -f "$source_dir/tests/ll_reference_profile/verify_installed.py" ]]; then
        printf '%s\n' 'Completed source/venv preparation is required.' >&2; return 2
    fi
    (
        cd -- "$source_dir" || return
        source scripts/ll_reference_env.sh || return
        unset SHMEM_ROOT CATLASS_ROOT DEEPEP_LL_STAGE_TIMER DEEPEP_LL_TIMER_PROFILE PYTHONPATH
        export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_BUILD_TYPE=Release DEEPEP_LL_REFERENCE_PROFILE=ON
        export DEEPEP_BUILD_MEGAMOE=OFF DEEPEP_MEGAMOE_TIMER=OFF DEEPEP_BUILD_EP_GMM_FUSED=OFF CCACHE_DISABLE=1
        if [[ $action != --skip-build ]]; then
            printf '%s\n' '[reference] Building one fresh SHMEM SDK and a complete reference wheel.'
            "$python_bin" scripts/build_shmem.py >"$group/logs/sdk-build.log" 2>&1 || return
            "$python_bin" scripts/clean_build_artifacts.py --wheel >"$group/logs/clean-wheel.log" 2>&1 || return
            "$python_bin" -m build --wheel --no-isolation --outdir "$group/wheels" >"$group/logs/wheel-build.log" 2>&1 || return
            local wheels=("$group"/wheels/*.whl)
            if (( ${#wheels[@]} != 1 )) || [[ ! -f ${wheels[0]} ]]; then printf '%s\n' 'Expected exactly one full wheel.' >&2; return 2; fi
            "$python_bin" -m pip install --no-deps --force-reinstall "${wheels[0]}" >"$group/logs/install.log" 2>&1 || return
            "$python_bin" -I tests/ll_reference_profile/verify_installed.py >"$group/build-ready.json" 2>"$group/logs/host-build.log" || return
        elif [[ ! -s "$group/build-ready.json" ]]; then
            printf '%s\n' 'Build is incomplete; --skip-build cannot sample.' >&2; return 2
        fi
        if [[ $action == --build-only ]]; then printf '%s\n' '[reference] Full build/Host checks complete; no NPU work ran.'; return 0; fi
        stamp="capture-$(date -u +%Y%m%dT%H%M%S)-$$"
        "$python_bin" -I tests/ll_reference_profile/verify_installed.py >"$group/logs/$stamp-host.json" 2>"$group/logs/$stamp-host.log" || return
        printf '%s\n' "[reference] Capturing $stamp; log: $group/logs/$stamp.log"
        "$python_bin" tests/ll_stage_timer/run.py \
            --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --backend mte \
            --num-processes 16 --num-nodes 1 --node-rank 0 \
            --master-addr "$address" --master-port "$port" --shmem-port "$shmem_port" \
            --num-tokens 1 2 4 8 16 24 32 --num-max-tokens-per-rank 32 \
            --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --fp8-scale-dtype int32 \
            --do-cpu-sync 0 --route normal --seed 0 --correctness-repeats 3 --repeat-combine 2 \
            --warmup 10 --num-iters 100 --perf-tail 50 --dispatch-only --ll-reference-profile --reference-iters 1 \
            --output-dir "$group/runs" --run-name "$stamp" "$@" >"$group/logs/$stamp.log" 2>&1 || return
        "$python_bin" tests/ll_reference_profile/analyze.py --input "$group/runs/$stamp" \
            --output-dir "$group/analysis/$stamp" --iteration last >"$group/logs/$stamp-analysis.log" 2>&1 || return
        printf '%s\n' "Return only: $group/analysis/$stamp/return_report.md" \
            "Full workbook stays on target: $group/analysis/$stamp/stage-times.xlsx"
    )
}
if reference_main "$@"; then
    :
else
    printf '%s\n' 'Reference workflow stopped. Preserve the group/logs; no automatic retries.' >&2
    false
fi
