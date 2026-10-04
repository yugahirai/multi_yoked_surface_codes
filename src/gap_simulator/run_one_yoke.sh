#!/usr/bin/env bash
# Run one simulator script (SCRIPT below) for several t_interval values and
# distances, stopping each run once the csv holds the wanted number of outer
# error events for that (t_interval, d). each simulator appends every run to
# the single csv data/simulated/<protocol>.csv at the repository root.
#
#   ./run_one_yoke.sh            # every t in T_INTERVALS x every d in DS
#   ./run_one_yoke.sh 9 10       # every t, only these d
#
# While it runs:
#   Ctrl+C once        kill the CURRENT simulation only (its rows so far are
#                      in the csv) and go on with the next (t, d)
#   Ctrl+C twice       (within 2 s) abort the whole job
#   kill -INT <pid>    same as Ctrl+C, from another terminal (pid is printed)
#
# Edit the settings block below. MAX_OUTER_ERRORS maps d -> outer error
# events to collect for every t_interval; an entry keyed "t:d" (e.g. [8:11]=100)
# overrides it for that one t_interval. Errors already in the csv from earlier
# runs count, so rerunning the job only tops up what is missing. --shots is
# just an upper bound (the run stops early at the target).
#
# sim_one_yoke.py, sim_two_yoke.py and sim_three_yoke.py all implement
# --max-outer-errors. The code they simulate is the MATRIX_PATH set inside
# each script; for the two- and three-yoke scripts put their extra options
# (e.g. --l1-check-rate 2) in EXTRA_ARGS.
set -uo pipefail
cd "$(dirname "$0")"

# --- settings ---------------------------------------------------------------
PY=../../.venv/bin/python
SCRIPT=sim_one_yoke.py
T_INTERVALS=(8)
P=0.001
SHOTS=100000000          # upper bound per (t, d)
PROCESSES=111
CHUNK_SIZE=1000
EXTRA_ARGS=(--trivial-first)   # e.g. (--trivial-first --batch-shots 32)

DS=(5 6 7 8 9 10 11)
declare -A MAX_OUTER_ERRORS=(
    [5]=1000
    [6]=1000
    [7]=1000
    [8]=1000
    [9]=1000
    [10]=300
    [11]=200
    # per-t_interval overrides, "t:d":
    # [8:11]=100
)
DOUBLE_INT_SEC=2   # second Ctrl+C within this many seconds aborts the job
# ---------------------------------------------------------------------------

if [ $# -gt 0 ]; then
    DS=("$@")
fi

target_for() {  # target_for <t> <d>
    local t=$1 d=$2
    if [ -n "${MAX_OUTER_ERRORS[$t:$d]+x}" ]; then
        echo "${MAX_OUTER_ERRORS[$t:$d]}"
    elif [ -n "${MAX_OUTER_ERRORS[$d]+x}" ]; then
        echo "${MAX_OUTER_ERRORS[$d]}"
    fi
}

# The simulation runs in its own process group (setsid), so a Ctrl+C on the
# terminal reaches only this script; the trap decides what to kill. Killing
# the group takes the multiprocessing workers down with the main process.
CHILD=""
LAST_INT=0
ABORT=0

kill_child() {  # kill_child <signal>
    if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
        kill "-$1" -- "-$CHILD" 2>/dev/null || kill "-$1" "$CHILD" 2>/dev/null
    fi
}

on_int() {
    local now
    now=$(date +%s)
    if (( now - LAST_INT <= DOUBLE_INT_SEC )); then
        echo
        echo ">>> second Ctrl+C: aborting the whole job"
        ABORT=1
        kill_child TERM
        return
    fi
    LAST_INT=$now
    echo
    echo ">>> Ctrl+C: killing the current simulation (t_interval=$t d=$d);" \
         "press Ctrl+C again within ${DOUBLE_INT_SEC}s to abort the whole job"
    kill_child TERM
}

on_term() {
    ABORT=1
    kill_child TERM
}

cleanup() {
    kill_child TERM
}

trap on_int INT
trap on_term TERM HUP
trap cleanup EXIT

echo "job pid $$  (kill -INT $$ skips the current simulation)"

for t in "${T_INTERVALS[@]}"; do
    for d in "${DS[@]}"; do
        [ "$ABORT" = 1 ] && break 2
        target=$(target_for "$t" "$d")
        if [ -z "$target" ]; then
            echo "no MAX_OUTER_ERRORS entry for d=$d, skipping" >&2
            continue
        fi
        echo "=== t_interval=$t  d=$d  target $target outer errors ==="
        setsid "$PY" "$SCRIPT" \
            --d "$d" \
            --p "$P" \
            --t-interval "$t" \
            --shots "$SHOTS" \
            --processes "$PROCESSES" \
            --chunk-size "$CHUNK_SIZE" \
            --max-outer-errors "$target" \
            "${EXTRA_ARGS[@]}" &
        CHILD=$!
        # wait returns early when a trapped signal arrives; keep waiting until
        # the simulation is really gone.
        status=0
        while kill -0 "$CHILD" 2>/dev/null; do
            wait "$CHILD"
            status=$?
        done
        CHILD=""
        if [ "$status" -ne 0 ]; then
            echo "=== simulation ended with status $status (t_interval=$t d=$d)"
        fi
    done
done

if [ "$ABORT" = 1 ]; then
    echo "job aborted"
    exit 130
fi
echo "job done"
