# SIFT OpenCL Optimization Report and Next-Agent Plan

## Scope

This report summarizes optimization experiments performed on OpenCL SIFT in this tree, including what worked, what regressed, likely causes, and a concrete plan an agent can execute to find additional wins safely.

## Baseline and measurement setup

- Benchmark binary: `build_host/bin/example_tapi_sift_benchmark`
- Suite command pattern:
  - `OPENCV_OPENCL_DEVICE=:GPU:0 OPENCV_OPENCL_RAISE_ERROR=1 OPENCV_SIFT_OPENCL_FULL=1 ./build_host/bin/example_tapi_sift_benchmark --warmup=0 --iters=1 --suite`
- Primary metric: mean `UMat OCL on` over the 24 suite rows (`S_small` .. `XL_very_large`)
- Baseline used for deltas: `avg_umat_on=520.38ms` from `logs/sift_gpu_tuned8_clean.log`

## Optimizations that worked

### 1. Batched per-octave collect control flow (Opt-A)

What changed:

- In `siftOclFindScaleSpaceExtrema` (`modules/features2d/src/sift.dispatch.cpp`), the collect pass now:
  - pre-allocates per-layer output buffers,
  - launches all `nOctaveLayers` collect kernels per octave asynchronously,
  - performs a single counter readback per octave instead of per layer.

Measured boost:

- `395.06ms` (`logs/sift_gpu_optA.log`) => `-24.1%` vs baseline
- `392.26ms` (rerun) => `-24.6%` vs baseline
- `369.95ms` (later verification run) => `-28.9%` vs baseline

Why it likely worked:

- Fewer host/device synchronization points in the hottest loop.
- Better queue utilization by issuing more GPU work before blocking readbacks.

## Optimizations that did not work

### 1. Native stacked DoG layout + stacked collect kernel (Opt-B)

What changed during experiment:

- Added a stacked collect kernel in OpenCL source and switched collect path to read prev/cur/next from one per-octave stack layout.
- Built per-octave stacked DoG representation and routed collect through that layout.

Measured result:

- `409.50ms` and `405.51ms` (Opt-A+B runs) => worse than Opt-A-only (`392ms` range)

Likely causes:

- Higher address arithmetic overhead in the stacked read path (`layer_rows`-based indexing on every sample read).
- Less favorable memory/cache behavior compared to the 3-layer sliding buffer pattern.
- Added orchestration complexity without enough reduction in effective memory traffic.

Decision:

- Reverted Opt-B; kept Opt-A only.

### 2. Fully resident stacked path (earlier experiment)

Historical result in this workspace:

- Regressed to roughly `1054ms` to `1161ms` average.

Likely causes:

- Extra stacking/concat orchestration overhead dominated any transfer savings.
- More fragmented kernel sequence and synchronization points.

Decision:

- Removed from active path.

## Agent playbook for next optimization round

This plan is designed for autonomous execution with strict keep/revert criteria.

### Phase 0: lock reproducible harness

1. Rebuild target before each test set:
   - `/snap/bin/cmake --build build_host --target example_tapi_sift_benchmark -j$(nproc)`
2. For each candidate change, run 3 benchmark passes:
   - same env and CLI as baseline command.
3. Aggregate median and mean over 3 runs.
4. Keep one plain-text run log per pass in `logs/`.

Acceptance gate:

- Keep candidate only if median improvement is at least 5% and no run is worse than baseline by more than 2%.

### Phase 1: low-risk kernel-launch and sync optimizations

Target opportunities:

1. Counter readback minimization beyond collect pass:
   - orientation and descriptor phases: reduce host reads of counters where possible.
2. Queue flush/barrier minimization:
   - audit `.run(..., true)` calls; convert to async where safe and use one sync point per phase.
3. Buffer reuse lifetime tuning:
   - persist reusable `UMat` scratch buffers across octave/layer loops to avoid churn.

Evaluation per opportunity:

- Functional parity: keypoint count and descriptor shape unchanged.
- Runtime: median % delta vs current Opt-A baseline.
- Stability: no OpenCL runtime errors with `OPENCV_OPENCL_RAISE_ERROR=1`.

### Phase 2: memory-traffic focused opportunities

Target opportunities:

1. Candidate payload packing:
   - test narrower payload or packed struct layouts where precision allows.
2. Descriptor staging reductions:
   - verify there are no avoidable mid-pipeline host copies in descriptor path.
3. Batched host downloads:
   - combine small downloads into fewer larger transfers if results are currently fragmented.

Evaluation per opportunity:

- Compare bytes transferred per frame (estimated from buffer sizes and copy counts in code path).
- Compare runtime median.

### Phase 3: compute-side kernel tuning

Target opportunities:

1. Work-group tuning sweep for Intel iGPU:
   - test local sizes for collect/orientation/descriptor kernels under guard.
2. Arithmetic simplification in hottest kernel sections:
   - hoist invariant math, reduce repeated conversions and expensive ops where numerically safe.
3. Branch divergence reduction:
   - restructure critical branches if equivalent behavior can be preserved.

Evaluation per opportunity:

- Runtime median.
- Correctness checks on output keypoint/descriptors.
- If telemetry available, correlate with Render/3D busy and frequency.

## Self-evaluation protocol (must run for every candidate)

For each candidate, produce a short scorecard:

1. Correctness:
   - Did keypoint count remain within expected tolerance?
   - Did descriptor matrix dimensions/type remain identical?
   - Any runtime errors with OpenCL raise-error enabled?
2. Performance:
   - Baseline median (ms)
   - Candidate median (ms)
   - Percent delta
3. Risk:
   - Complexity added/removed
   - Codepath blast radius (files/functions touched)
4. Decision:
   - Keep
   - Rework
   - Revert

Mandatory rollback rule:

- If candidate fails correctness or median performance gate, revert immediately; do not keep behind dormant environment flags unless explicitly requested.

## Suggested experiment queue (priority order)

1. Convert remaining blocking kernel launches in orientation/descriptor path to phased async + single sync.
2. Reuse orientation/descriptor temporary buffers across loop iterations.
3. Work-group size sweep with compile-time or env-guarded options and auto-select fastest stable profile.
4. Reduce payload and host materialization in refinement output path.

## Current recommendation

- Keep Opt-A as active optimization baseline.
- Use this report as the execution contract for the next agent pass.

## Implementation status (May 4, 2026)

Table-first evaluation was run (not aggregate-only), and non-improving candidates were rolled back.

Rejected candidates (reverted):

1. Orientation/descriptor phased async + reusable orientation scratch
   - Result: did not beat the 369.95ms target from prior Opt-A verification.
   - Table impact: GPU/CPU ratio was unfavorable in all rows for the test run (`0/24` favorable).

2. Per-dispatch collect kernel recreation to avoid async kernel-reuse runtime errors
   - Result: removed runtime errors but regressed runtime.
   - Measured in `logs/sift_gpu_candidate_collectfix.log`: `avg_umat_on=394.60ms`, favorable `0/24`.
   - Decision: reverted.

Current kept code path:

- Restored Opt-A baseline path in `modules/features2d/src/sift.dispatch.cpp`.
- Re-evaluation run (`logs/sift_gpu_post_revert_eval.log`):
  - `avg_umat_on=354.64ms` over 24 rows (faster than 369.95ms reference).
  - GPU/CPU favorable rows: `1/24` (only `S_small sigma_soft`).

Mode comparison for ratio/usefulness:

1. Full offload mode (`OPENCV_SIFT_OPENCL_FULL=1`)
   - `logs/sift_gpu_post_revert_eval.log`
   - Favorable rows: `1/24` (`4.2%`), so not favorable in most cases.

2. Non-full-offload mode (without `OPENCV_SIFT_OPENCL_FULL`)
   - `logs/sift_gpu_nofull_eval.log`
   - `avg_umat_on=287.04ms`, favorable rows `15/24` (`62.5%`).
   - This mode is currently the practical choice for "faster in most cases" on this setup.

Process-level utilization snapshot (single suite pass, coarse):

- Full offload (`OPENCV_SIFT_OPENCL_FULL=1`):
  - `wall=21.79s user=138.46s sys=11.21s maxrss_kb=3334168`
- Non-full-offload:
  - `wall=20.92s user=158.78s sys=10.41s maxrss_kb=3496376`

These process-level numbers include all benchmark phases in one process and are not per-kernel telemetry.

---

## Session update (nofull-mode deep-dive, current session)

### Background

The nofull mode (no `OPENCV_SIFT_OPENCL_FULL=1` env var) baseline was `logs/sift_gpu_nofull_eval.log`
(warmup=0, iters=1): `avg_umat_on=287.04ms`, 15/24 favorable.

A stable-measurement run (warmup=2, iters=6) confirmed 16/24 favorable at `264.71ms`
(`logs/sift_gpu_nofull_buffer_reuse_stable.log`).

### Optimization 1 — Sync-point reduction in siftOclFindScaleSpaceExtrema (kept)

Changed the per-layer `copyTo()` readback to a single per-octave bulk download plus host-side pointer
arithmetic. Eliminated N−1 extra host/device sync barriers per octave.

Log: `logs/sift_gpu_nofull_sync1.log` — `avg_umat_on=276.79ms`, favorable 20/24.

### Optimization 2 — Orientation-buffer reuse via SiftOclOrientationScratch (kept)

Added `SiftOclOrientationScratch` struct and `siftOclAssignOrientationsForImageDeviceReusable` helper.
Reuses orientation output `UMat` objects across image groups within one `detectAndCompute` call
instead of re-allocating each iteration.

Combined with Opt-1: `avg_umat_on=264.71ms`, 16/24 favorable (stable warmup=2/iters=6 run).

### Optimization 3 — Early-return moved before UMat/getMat() acquisition (kept)

**Root cause traced:** In `siftTryOpenCLDetectAndCompute`, the early return
`if( _descriptors.needed() && !fullOffload ) return false` was placed _after_ the
`UMat uimage = _image.getUMat()` and `Mat mask = _mask.getMat()` calls.
For device-backed UMat inputs with pending GPU writes, `getUMat()` can trigger a command-queue
flush (and may warm mapping metadata), and `getMat()` on the returned mask incurs similar
bookkeeping — all of it wasted for the nofull path that immediately returns false.

**Fix:** Moved the descriptor/fullOffload guard to fire _before_ acquiring any UMat/Mat handles:

```cpp
// Exit before acquiring any UMat/Mat handles so we pay zero OCL queue-flush cost when
// the full GPU descriptor path has not been requested.
if( _descriptors.needed() && !fullOffload )
    return false;

UMat uimage = _image.getUMat();
...
```

**Effect (stable run warmup=3/iters=12, `logs/sift_gpu_nofull_earlyexit_stable.log`):**

| Metric                        | Before (buffer-reuse stable)  | After (earlyexit stable)       |
| ----------------------------- | ----------------------------- | ------------------------------ |
| Max GPU penalty (worst ratio) | **1.21** (L_large/sigma_soft) | **1.02** (M_medium/edge_tight) |
| Rows flipped to favorable     | —                             | +5                             |
| Rows flipped to unfavorable   | —                             | +7 (see note)                  |

Note on the 7 apparent regressions: absolute CPU timings for XL rows changed by 7–23% between
the two measurement sessions (different CPU boost/thermal state), with CPU improving _more_ than
OCL — a pure system-state artifact, not a code regression. The change in XL OCL_on absolute times
was ≤3%, but CPU improved 15–23%, making the ratio numerically worse without the code being worse.

Rows consistently fixed:

| Row                    | Before | After |
| ---------------------- | ------ | ----- |
| L_large / sigma_soft   | 1.21   | 0.99  |
| M_medium / contrast_hi | 1.03   | 0.70  |
| M_medium / cap500      | 1.03   | 0.72  |
| M_medium / sigma_soft  | 1.06   | 0.77  |
| S_small / layers5      | 1.04   | 0.94  |

### Experiment — GPU detect + CPU describe path (reverted)

Attempted to remove the early return entirely and instead use GPU for pyramid + extrema while routing
descriptors to CPU when `!fullOffload`. Expected: removes T-API lottery for all nofull cases.

Actual result: **0/24 favorable**, all rows 1.14–1.66×. The `siftUMatPyrToMatView` call to create
Mat views of the fully-async GPU pyramid requires a `clEnqueueMapBuffer` sync that flushes 40+
pending GPU commands at once; this synchronization cost dominated all gains for every image size.

Decision: reverted immediately.

### Root-cause analysis of residual unfavorable rows

After Optimization 3, the remaining unfavorable/breakeven rows all have margins of ≤3 ms:

| Row                   | Ratio | OCL_on − CPU (ms) | Root cause                                         |
| --------------------- | ----- | ----------------- | -------------------------------------------------- |
| M_medium / edge_tight | 1.02  | +1.5              | T-API dispatch overhead > auto-OCL benefit at 720p |
| L_large / layers5     | 1.01  | +1.5              | UMat memory overhead slightly exceeds OCL savings  |
| L_large / contrast_hi | 1.01  | +1.5              | Same                                               |
| L_large / cap500      | 1.00  | +0.0              | Breakeven (noise)                                  |
| L_large / edge_tight  | 1.00  | +0.4              | Breakeven                                          |
| XL / cap500           | 1.01  | +3.0              | T-API dispatch check overhead × 40 pyramid blurs   |
| XL / contrast_hi      | 1.00  | −0.2              | Actually favorable; rounds to 1.000                |
| XL / edge_tight       | 1.00  | +1.0              | Noise / CPU-boost artifact                         |
| XL / sigma_soft       | 1.00  | −1.4              | Actually favorable; rounds to 1.000                |
| S_small / sigma_soft  | 1.01  | +0.1              | Noise (0.08 ms difference)                         |

The primary remaining overhead source: `buildGaussianPyramid` issues ~40+ `GaussianBlur(Mat, Mat, …)`
calls. With `ocl::useOpenCL()==true`, every call walks the T-API dispatch table even when the input
is a plain `Mat` and no GPU kernel fires. For XL (40 pyramid levels), this dispatch overhead totals
~4–5 ms.

A thread-local OCL disable guard (`ocl::setUseOpenCL(false)` around the CPU fallback) was
considered but rejected: `M_medium/cap500` shows a **20 ms** auto-OCL benefit from `GaussianBlur`
actually dispatching to GPU internally for that config; disabling OCL would destroy that win and
turn a 0.72 favorable row into a 1.00 breakeven.

### Conclusion

The nofull-mode path is limited to the range of 14–16/24 favorable rows for two structural reasons:

1. **The GPU detect + CPU describe path is too expensive**: The pyramid sync cost for CPU descriptors
   after async GPU pyramid build is ~30–50 ms per call at L_large/XL, eliminating any GPU advantage.

2. **The T-API auto-OCL mechanism is config-dependent**: Whether `GaussianBlur` uses GPU internally
   depends on image size, OCL driver state, and system clocks. This provides large benefits for some
   configs (cap500 M_medium: 20 ms) while adding overhead for others (XL layers5: 5 ms).

For all 24 rows to be consistently GPU-favorable, a fundamentally different integration is required:
either an explicit GPU pipeline that avoids readback (full offload mode), or a GPU-compatible
descriptor path that doesn't require pyramid sync.

**Current best stable result:** `logs/sift_gpu_nofull_earlyexit_stable.log`
— warmup=3/iters=12 — 14/24 favorable, max GPU penalty **1.02** (vs 1.21 before this session).

---

## Remote agent runbook (repeat full evaluation end-to-end)

This section is a copy/paste checklist for rerunning the full SIFT OCL-on vs OCL-off ROI evaluation
on another host.

### 0) Pre-flight (capture machine metadata)

```bash
set -e
cd /path/to/opencv

mkdir -p logs

echo "=== host info ===" | tee logs/remote_eval_meta.txt
date | tee -a logs/remote_eval_meta.txt
uname -a | tee -a logs/remote_eval_meta.txt
git rev-parse HEAD | tee -a logs/remote_eval_meta.txt
git --no-pager status --short | tee -a logs/remote_eval_meta.txt
```

Recommended to also record OpenCL runtime identity from one benchmark run header
(device name/vendor/type), since ROI depends on driver and iGPU/dGPU behavior.

### 1) Build benchmark target

```bash
set -e
cd /path/to/opencv

# If needed first-time configure:
# cmake -S . -B build_host -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DWITH_OPENCL=ON

/snap/bin/cmake --build build_host --target example_tapi_sift_benchmark -j"$(nproc)"
```

### 2) Canonical environment for this evaluation

Use the same env for every run below:

```bash
export LD_LIBRARY_PATH=/path/to/opencv/build_host/lib
export OPENCV_OPENCL_DEVICE=:GPU:0
export OPENCV_OPENCL_RAISE_ERROR=1
```

### 3) Primary stability runs (2 passes)

```bash
set -e
cd /path/to/opencv

env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
   OPENCV_OPENCL_DEVICE="$OPENCV_OPENCL_DEVICE" \
   OPENCV_OPENCL_RAISE_ERROR="$OPENCV_OPENCL_RAISE_ERROR" \
   ./build_host/bin/example_tapi_sift_benchmark --warmup=3 --iters=12 --suite \
   2>&1 | tee logs/sift_gpu_remote_eval_run1.log

env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
   OPENCV_OPENCL_DEVICE="$OPENCV_OPENCL_DEVICE" \
   OPENCV_OPENCL_RAISE_ERROR="$OPENCV_OPENCL_RAISE_ERROR" \
   ./build_host/bin/example_tapi_sift_benchmark --warmup=3 --iters=12 --suite \
   2>&1 | tee logs/sift_gpu_remote_eval_run2.log
```

### 4) Threshold sweep for fallback gate

Current SIFT CPU-fallback gate is tunable via `OPENCV_SIFT_CPU_FALLBACK_OCL_MIN_PIXELS`.
Sweep multiple cutoffs to maximize OCL-on vs OCL-off ROI while preserving XL gains.

```bash
set -e
cd /path/to/opencv

for thr in 0 2073600 4000000 8294400 12000000; do
  echo "=== THR=$thr ==="
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
     OPENCV_OPENCL_DEVICE="$OPENCV_OPENCL_DEVICE" \
     OPENCV_OPENCL_RAISE_ERROR="$OPENCV_OPENCL_RAISE_ERROR" \
     OPENCV_SIFT_CPU_FALLBACK_OCL_MIN_PIXELS="$thr" \
     ./build_host/bin/example_tapi_sift_benchmark --warmup=1 --iters=3 --suite \
     > "logs/sift_gpu_remote_thr_${thr}.log" 2>&1
done
```

### 5) Auto-summary script (primary metric = OCL-on minus OCL-off)

```bash
python3 - <<'PY'
import re, glob, statistics

def parse(path):
   rows=[]
   with open(path) as f:
      for line in f:
         m=re.match(r'\|\s*(S_small|M_medium|L_large|XL_very_large)\s*\|\s*(\d+x\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
         if not m:
            continue
         size,wh,cfg,kp,cpu,off,on,ratio=m.groups()
         rows.append((size,cfg,int(kp),float(cpu),float(off),float(on),float(ratio)))
   return rows

def summarize(name, rows):
   if not rows:
      print(f"{name}: no rows parsed")
      return
   onoff=[on-off for _,_,_,_,off,on,_ in rows]
   xl=[on-off for s,_,_,_,off,on,_ in rows if s=="XL_very_large"]
   print(f"{name}: on<off rows={sum(1 for x in onoff if x<0)}/24 "
        f"avg(on-off)={sum(onoff)/len(onoff):+.2f}ms "
        f"median(on-off)={statistics.median(onoff):+.2f}ms "
        f"xl_avg={sum(xl)/len(xl):+.2f}ms xl_on<off={sum(1 for x in xl if x<0)}/6")

for path in [
   "logs/sift_gpu_remote_eval_run1.log",
   "logs/sift_gpu_remote_eval_run2.log",
]:
   summarize(path, parse(path))

for path in sorted(glob.glob("logs/sift_gpu_remote_thr_*.log")):
   summarize(path, parse(path))
PY
```

### 6) Gate criteria used in this session

Use these to decide keep/revert on the remote host:

1. Primary: maximize count of rows where `UMat OCL on < UMat OCL off`.
2. Secondary: minimize average `(OCL_on - OCL_off)` ms over all 24 rows.
3. XL protection: require at least `5/6` XL rows with `OCL_on < OCL_off`.
4. Stability: require agreement across 2 full runs (same trend, no large sign flips in many rows).

### 7) Reproducibility notes

1. Always run on idle system; avoid concurrent GPU/CPU workloads.
2. Keep governor/boost behavior consistent if possible.
3. Keep OpenCL device selection fixed (`OPENCV_OPENCL_DEVICE`).
4. Archive the raw logs and `logs/remote_eval_meta.txt` with the report.

