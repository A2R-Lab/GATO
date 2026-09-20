#!/usr/bin/env bash
# Shared shell for the timing-night drivers (source it; do not execute).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/night_lib.sh"
#
# Callers set LOGDIR (per-leg logs) and SUMMARY (the PASS/FAIL digest file)
# before calling stage/leg. Everything here is the box-etiquette every night
# runner used to carry its own copy of: the GPU queries, the quiet-box
# preflight, the settle-wait for queue stragglers, and the per-leg wrapper
# with rc capture + the orphan fence (nothing of ours may outlive its leg —
# pipes/timeouts orphan children, and the GPU must be EMPTY between timing
# legs).

night_gpu_apps() {  # number of compute processes on the GPU (0 = idle)
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true
}

night_gpu_util() {  # GPU utilization percent
  local u
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "${u:-0}"
}

night_preflight() {  # abort (rc 1) unless the box is quiet; FORCE=1 overrides (plumbing smoke only)
  [[ "${FORCE:-0}" == "1" ]] && return 0
  local util apps load
  util=$(night_gpu_util); apps=$(night_gpu_apps); load=$(awk '{print int($1)}' /proc/loadavg)
  if (( util > 5 || apps > 0 || load > 2 )); then
    echo "ABORT: box not quiet (gpu_util=${util}% compute_apps=${apps} load=${load})." \
         "FORCE=1 overrides (plumbing smoke only)." | tee -a "${SUMMARY:-/dev/null}"
    return 1
  fi
  return 0
}

night_settle() {  # wait up to $1 seconds for the GPU to clear (queue stragglers); rc 1 if still busy
  local limit=${1:-600} waited=0 apps util
  (( limit > 0 )) || return 0
  while (( waited < limit )); do
    apps=$(night_gpu_apps); util=$(night_gpu_util)
    (( apps == 0 && util <= 5 )) && break
    (( waited == 0 )) && echo "waiting for the GPU to clear (apps=$apps util=${util}%)..."
    sleep 20; waited=$((waited + 20))
  done
  apps=$(night_gpu_apps)
  if (( apps > 0 )); then
    echo "ABORT: GPU still busy after ${limit}s — timing needs a SOLO slot. Nothing was run." >&2
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >&2
    return 1
  fi
  (( waited > 0 )) && echo "GPU clear after ${waited}s."
  return 0
}

stage() {  # stage <name> <log> <cmd...> — run one leg with an explicit log path
  local name=$1 log=$2; shift 2
  local t0=$SECONDS rc=0
  echo "==== [$name] $(date -u +%H:%M:%S) $*" | tee -a "$SUMMARY"
  if "$@" >"$log" 2>&1; then
    echo "[$name] PASS  ($(( (SECONDS-t0)/60 )) min)  log=$log" | tee -a "$SUMMARY"
  else
    rc=$?
    echo "[$name] FAIL rc=$rc ($(( (SECONDS-t0)/60 )) min)  log=$log  <-- review" | tee -a "$SUMMARY"
  fi
  # orphan fence: nothing of ours may outlive its leg
  local orph
  orph=$(night_gpu_apps)
  if (( orph > 0 )); then
    echo "[$name] WARNING: $orph compute app(s) still on the GPU post-leg" | tee -a "$SUMMARY"
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader >> "$SUMMARY"
  fi
  return $rc
}

leg() {  # leg <name> <cmd...> — stage with the log at $LOGDIR/<name>.log
  local name=$1; shift
  stage "$name" "$LOGDIR/$name.log" "$@"
}
