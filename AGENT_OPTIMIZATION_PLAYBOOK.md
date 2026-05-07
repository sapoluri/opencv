# SIFT GPU Optimization: Agent Continuation Playbook

**Purpose:** Enable an autonomous agent to continue GPU optimization work on new hardware, aware of baseline results and expected performance scaling.

**Reference Document:** `SIFT_GPU_OPTIMIZATION_ANALYSIS.md` (contains detailed baseline, all phases, root causes)

---

## Stage 0: Hardware Characterization (MUST RUN FIRST)

### Step 0.1: Identify Target Hardware

Collect these facts **before** running any benchmarks:

```bash
#!/bin/bash
echo "=== GPU Identification ==="
lspci | grep -i "3d\|vga"
clinfo | grep -A3 "Device Name"
clinfo | grep -A3 "Max Compute Units"
clinfo | grep -A3 "Max Work Group Size"

echo "=== Driver Version ==="
cat /sys/module/i915/version 2>/dev/null || echo "i915 not loaded"
cat /sys/module/xe/version 2>/dev/null || echo "xe not loaded"
clinfo | grep -A1 "Device Version"

echo "=== System Memory ==="
free -h | grep Mem
```

### Step 0.2: Classify GPU Category

Determine which category your GPU falls into. This determines expected performance scaling:

| Category | Examples | Expected Vs Baseline | Key Characteristics |
|----------|----------|---|---|
| **Integrated (iGPU)** | Ultra 7 255H + Intel Arc, Intel Core Ultra, Apple M-series | +2–4% | Shared memory, 16–24 EU, <100 GB/s bandwidth |
| **Entry dGPU** | Intel Arc A380, NVIDIA RTX 4060 | +5–8% | 64–128 EU, discrete VRAM, 250+ GB/s bandwidth |
| **Mid-range dGPU** | Intel Arc A770, NVIDIA RTX 4070 | +8–15% | 256–384 EU, 500+ GB/s bandwidth, large caches |
| **High-end dGPU** | NVIDIA RTX 4090, AMD MI300 | +15–25% | 1000+ EU, 900+ GB/s bandwidth, massive caches |
| **Data Center GPU** | Intel Arc A50, NVIDIA A100 | +12–20% | Optimized for compute, relaxed graphics constraints |

**Decision Flow:**
```
if GPU_COMPUTE_UNITS < 64:
    category = "Integrated (iGPU)"
elif GPU_BANDWIDTH_GB_S < 300:
    category = "Entry dGPU"
elif GPU_BANDWIDTH_GB_S < 600:
    category = "Mid-range dGPU"
else:
    category = "High-end dGPU"
```

### Step 0.3: Baseline Hardware (Ultra 7 255H + Intel Arc - Reference)

Compare your GPU to the baseline:

**Baseline (Ultra 7 255H + Intel Arc iGPU) Specs:**
- **Compute Units:** 128 EU
- **Subgroup Size:** 16 elements (rigid)
- **Max Clock:** 2250 MHz
- **Memory Type:** Shared DDR5 with CPU
- **Memory Bandwidth:** ~76 GB/s
- **L3 Cache:** Shared CPU/GPU, ~12 MB
- **Baseline Performance:** 0.963× GPU/CPU (14/24 favorable)

**Comparison Table Template:**

| Metric | Baseline (Ultra 7 255H + Intel Arc) | Your Hardware | Ratio | Implication |
|---|---|---|---|---|
| **Compute Units** | 128 | ? | ? | More EU = more parallelism |
| **Max Frequency (MHz)** | 2250 | ? | ? | Frequency scaling @ 50% improvement per 1GHz |
| **Memory Bandwidth (GB/s)** | 76 | ? | ? | More BW = better for memory-bound kernels |
| **Subgroup Size** | 16 | ? | ? | >16 may need geometry re-tuning |
| **L3/SLM Cache (MB)** | ~12 shared | ? | ? | More cache = better for SIFT histograms |

**Fill in the ? values using:**
```bash
# Compute Units
clinfo | grep "Max Compute Units"

# Max Frequency
clinfo | grep "Max Clock"

# Memory Bandwidth (estimate from NVIDIA/AMD specs or calculate from bus width × freq)
# For rough estimate: 256-bit × 2.5 GHz / 8 = 80 GB/s

# Subgroup Size
clinfo | grep "Preferred Work Group Multiple"

# Cache
clinfo | grep "Global Memory Cache Size" or specs sheet
```

---

## Stage 1: Establish Baseline on New Hardware

### Step 1.1: Build and Verify

```bash
set -e
cd /path/to/opencv

# Clean build
rm -rf build_host
mkdir -p build_host
cd build_host
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL=ON -DBUILD_EXAMPLES=ON ..
make -j$(nproc)

# Verify binary exists
test -f bin/example_tapi_sift_benchmark && echo "✓ Binary built"
```

### Step 1.2: Run Baseline Benchmark (Phases 1-3 Already Active)

```bash
#!/bin/bash
set -e
cd /path/to/opencv
export LD_LIBRARY_PATH="$PWD/build_host/lib:${LD_LIBRARY_PATH}"
export OPENCV_OPENCL_DEVICE=":GPU:0"
export OPENCV_OPENCL_RAISE_ERROR=1

mkdir -p logs

# Warmup=3, iters=12 for stable measurement
./build_host/bin/example_tapi_sift_benchmark --warmup=3 --iters=12 --suite \
  2>&1 | tee logs/baseline_new_hw.log

echo ""
echo "=== ANALYSIS ==="
python3 << 'PYSCRIPT'
import re
with open("logs/baseline_new_hw.log") as f:
    lines = f.readlines()
    
favorable = 0
total = 0
ratios = []
for line in lines:
    m = re.match(r'\|\s*(S_small|M_medium|L_large|XL_very_large).*\|\s*(\d+\.\d+)\s*\|', line)
    if m:
        ratio = float(m.group(2))
        ratios.append(ratio)
        if ratio < 1.0:
            favorable += 1
        total += 1

if ratios:
    avg = sum(ratios) / len(ratios)
    print(f"Favorable: {favorable}/{total} ({100*favorable/total:.1f}%)")
    print(f"Average GPU/CPU Ratio: {avg:.4f}×")
    print(f"Baseline Ultra 7 255H + Intel Arc: 0.9630× (14/24 favorable)")
    delta_pct = (avg - 0.963) / 0.963 * 100
    print(f"Delta vs Baseline: {delta_pct:+.2f}%")
PYSCRIPT
```

### Step 1.3: Interpret Baseline Results

**Expected Outcomes by GPU Category:**

#### Integrated iGPU (Ultra 7 255H + Intel Arc-like)
- **Expected:** 0.963× ± 0.020× (same as baseline)
- **14/24 favorable expected**
- **Action:** If within ±2%, you have a good match; proceed to Stage 2
- **Action:** If >3% worse, check OpenCL driver version and compiler

#### Entry dGPU
- **Expected:** 0.920–0.950× (5–8% faster than baseline)
- **16–18/24 favorable expected**
- **Action:** If achieved, Phases 1-3 are well-tuned; try Phase 5c
- **Action:** If worse, geometry may need re-tuning for larger subgroups

#### Mid-range dGPU
- **Expected:** 0.880–0.920× (8–15% faster)
- **18–20/24 favorable expected**
- **Action:** Consider re-tuning work-group geometry
- **Action:** Phase 5c may yield +1–3% additional gain

#### High-end dGPU
- **Expected:** 0.800–0.880× (15–25% faster)
- **20–22/24 favorable expected**
- **Action:** Consider kernel-level optimizations
- **Action:** Phase 4a/5d may succeed on this hardware

### Step 1.4: Document and Compare

**Create a comparison report:**

```bash
cat > logs/hw_comparison.md << 'REPORT'
# Hardware Comparison Report

## Target Hardware
- GPU: [Your GPU Model]
- Compute Units: [X]
- Memory Bandwidth: [Y] GB/s
- Category: [iGPU / Entry dGPU / Mid-range dGPU / High-end dGPU]

## Baseline Performance

### Ultra 7 255H + Intel Arc Reference (from SIFT_GPU_OPTIMIZATION_ANALYSIS.md)
- Favorable: 14/24 (58.3%)
- Average Ratio: 0.9630×
- Worst Case: S_small 1.107×
- Best Case: XL_very_large 0.920×

### New Hardware (This Run)
- Favorable: [X]/24
- Average Ratio: [Y]×
- Worst Case: [...]
- Best Case: [...]

## Performance Delta
- Ratio Delta: [+X%] vs baseline
- Favorable Delta: [+X] rows vs baseline
- GPU Category Expectation Met: [YES/NO]

## Next Steps
- [ ] If within expected range: Proceed to Stage 2 optimization
- [ ] If worse than expected: Debug driver/compiler
- [ ] If better than expected: Consider kernel-level optimizations

REPORT
cat logs/hw_comparison.md
```

---

## Stage 2: Tuning & Optimization Strategy

### Step 2.1: Decision Tree

```
IF baseline >= expected_for_category:
    → PHASE A: Safe environment variable tuning (5–10 min)
    → PHASE B: Kernel-level optimization (depends on category)
    → PHASE C: Regression testing & finalization

ELSE IF baseline < expected_for_category by >3%:
    → DEBUG: Check OpenCL driver, compiler version
    → RETRY: Run baseline again to rule out thermal throttling
    → ESCALATE: Check OPENCV_OPENCL_RAISE_ERROR for warnings
```

### Step 2.2: Phase A — Environment Variable Sweep (Low Risk)

**Rationale:** On Ultra 7 255H + Intel Arc, this found no improvement. On discrete GPUs with larger queues, it may help.

**Test Matrix:**

```bash
#!/bin/bash
set -e
cd /path/to/opencv
export LD_LIBRARY_PATH="$PWD/build_host/lib:${LD_LIBRARY_PATH}"
export OPENCV_OPENCL_DEVICE=":GPU:0"
export OPENCV_OPENCL_RAISE_ERROR=1

for ori_local in 32 48 64 96 128 256; do
  for desc_local in 64 96 128 256; do
    echo "=== ORI_LOCAL=$ori_local DESC_LOCAL=$desc_local ==="
    export OPENCV_SIFT_OCL_ORI_LOCAL=$ori_local
    export OPENCV_SIFT_OCL_DESC_LOCAL=$desc_local
    timeout 300 ./build_host/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite \
      2>&1 | tee "logs/sweep_ori${ori_local}_desc${desc_local}.log" | tail -2
  done
done

echo ""
echo "=== BEST CONFIGURATION ==="
python3 << 'PYSCRIPT'
import re, glob
best_ratio = float('inf')
best_config = None
for log_file in glob.glob("logs/sweep_ori*.log"):
    m = re.search(r'ori(\d+)_desc(\d+)', log_file)
    if not m:
        continue
    ori, desc = m.groups()
    with open(log_file) as f:
        text = f.read()
        ratios = re.findall(r'\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', text)
        if ratios:
            avg = sum(float(r) for r in ratios) / len(ratios)
            if avg < best_ratio:
                best_ratio = avg
                best_config = (ori, desc, avg)

if best_config:
    ori, desc, ratio = best_config
    print(f"✓ Best: ORI_LOCAL={ori} DESC_LOCAL={desc}")
    print(f"  Average GPU/CPU: {ratio:.4f}×")
    baseline = 0.963
    delta = (ratio - baseline) / baseline * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    if delta > -0.5:
        print(f"  ⚠ No improvement found; using defaults")
    else:
        print(f"  ✓ Improvement found; recommend setting env vars")
PYSCRIPT
```

**Gate:** Keep tuned values ONLY if average ratio improves by >0.5% vs baseline.

### Step 2.3: Phase B — Kernel-Level Optimization (Category-Dependent)

**IMPORTANT:** Only attempt if:
1. Baseline is better than expected (>5% faster than Ultra 7 255H + Intel Arc)
2. Phase A tuning yielded no improvement
3. GPU category is dGPU (discrete) with >256 EU

#### Option B1: Phase 5C — Orientation Batching (Discrete GPU only)

**Risk Level:** MEDIUM (regressed on Ultra 7 255H + Intel Arc; may help on discrete)

**Condition to Attempt:**
```
IF gpu_category IN ["Mid-range dGPU", "High-end dGPU"]:
    TRY phase_5c()
ELSE:
    SKIP (likely to regress)
```

**Implementation Sketch:**
1. Clone current `sift.dispatch.cpp` to backup
2. Apply Phase 5C changes (see SIFT_GPU_OPTIMIZATION_ANALYSIS.md Phase 5C section)
3. Rebuild and benchmark 24-case suite
4. **Gate:** Only keep if >1% improvement AND no regressions on XL

#### Option B2: Phase 5D — Cooperative Kernels (High-end dGPU only)

**Risk Level:** HIGH (regressed on Ultra 7 255H + Intel Arc; very high risk)

**Condition to Attempt:**
```
IF gpu_category == "High-end dGPU" AND baseline_ratio < 0.85:
    CAUTIOUSLY_TRY phase_5d()
ELSE:
    SKIP (risk/reward not favorable)
```

**Pre-Attempt Checklist:**
- [ ] Baseline GPU/CPU is <0.85× (>18% faster than baseline)
- [ ] Orientation kernel is confirmed as bottleneck (use profiler)
- [ ] Have backed up current code state
- [ ] Have 30 minutes for debug if regression occurs

#### Option B3: Phase 4A — Kernel Pragmas (NOT RECOMMENDED)

**Skip entirely.** These regressed on Ultra 7 255H + Intel Arc and are extremely compiler-dependent. The vendor (Intel IGC, LLVM) already performs these optimizations.

---

## Stage 3: Validation & Regression Testing

### Step 3.1: Correctness Verification

After any optimization, verify correctness:

```bash
#!/bin/bash
cd /path/to/opencv

# Rebuild
cmake --build build_host --target opencv_features2d example_tapi_sift_benchmark -j$(nproc)

# Check for errors
if [ $? -ne 0 ]; then
    echo "❌ BUILD FAILED"
    exit 1
fi

# Run full suite with error checking
export OPENCV_OPENCL_RAISE_ERROR=1
export LD_LIBRARY_PATH="$PWD/build_host/lib:${LD_LIBRARY_PATH}"
export OPENCV_OPENCL_DEVICE=":GPU:0"

timeout 300 ./build_host/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite \
  2>&1 | tee /tmp/correctness_check.log

# Parse for errors
if grep -i "error\|exception\|abort" /tmp/correctness_check.log; then
    echo "❌ RUNTIME ERRORS DETECTED"
    exit 1
fi

echo "✓ No runtime errors detected"

# Verify output count is reasonable (sanity check)
python3 << 'PYSCRIPT'
import re
with open("/tmp/correctness_check.log") as f:
    lines = f.readlines()
    
keypoint_counts = []
for line in lines:
    m = re.match(r'\|.*\|\s*(\d+)\s*\|', line)
    if m:
        kp = int(m.group(1))
        keypoint_counts.append(kp)

if keypoint_counts:
    min_kp = min(keypoint_counts)
    max_kp = max(keypoint_counts)
    print(f"Keypoint count range: {min_kp}–{max_kp}")
    
    # Sanity checks
    if min_kp < 100:
        print(f"⚠ WARNING: Very low keypoint count {min_kp}")
    if max_kp < 100000:
        print(f"⚠ WARNING: Low max keypoint count {max_kp}")
    else:
        print(f"✓ Keypoint counts in expected range")
PYSCRIPT
```

### Step 3.2: Performance Regression Test

**Acceptance Criteria:**
- Average GPU/CPU ratio must not worsen by >1% from baseline on new HW
- XL images must remain >80% favorable (5+/6)
- No individual case worse than +5% vs baseline

```bash
#!/bin/bash
python3 << 'PYSCRIPT'
import re

baseline_log = "logs/baseline_new_hw.log"
optimized_log = "logs/optimized_new_hw.log"  # After optimization

def parse_results(log_file):
    results = {}
    with open(log_file) as f:
        for line in f:
            m = re.match(r'\|\s*(S_small|M_medium|L_large|XL_very_large).*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', line)
            if m:
                size, cfg, kp, ratio = m.groups()
                results[f"{size}_{cfg}"] = float(ratio)
    return results

baseline = parse_results(baseline_log)
optimized = parse_results(optimized_log)

print("=== REGRESSION TEST ===\n")

if not baseline or not optimized:
    print("ERROR: Could not parse log files")
    exit(1)

regressions = 0
improvements = 0
xl_favorable = 0
xl_total = 0

for key in baseline:
    if key not in optimized:
        print(f"WARNING: {key} missing from optimized results")
        continue
    
    b_ratio = baseline[key]
    o_ratio = optimized[key]
    delta_pct = (o_ratio - b_ratio) / b_ratio * 100
    
    if "XL" in key:
        xl_total += 1
        if o_ratio < 1.0:
            xl_favorable += 1
    
    if delta_pct > 5.0:  # >5% worse
        print(f"❌ REGRESSION: {key}: {b_ratio:.4f}× → {o_ratio:.4f}× ({delta_pct:+.1f}%)")
        regressions += 1
    elif delta_pct < -1.0:  # >1% better
        print(f"✓ Improved: {key}: {b_ratio:.4f}× → {o_ratio:.4f}× ({delta_pct:+.1f}%)")
        improvements += 1

print(f"\n=== SUMMARY ===")
b_avg = sum(baseline.values()) / len(baseline)
o_avg = sum(optimized.values()) / len(optimized)
overall_delta = (o_avg - b_avg) / b_avg * 100

print(f"Regressions (>5%): {regressions}")
print(f"Improvements (>1%): {improvements}")
print(f"XL Favorable: {xl_favorable}/{xl_total}")
print(f"Average GPU/CPU Ratio: {b_avg:.4f}× → {o_avg:.4f}× ({overall_delta:+.2f}%)")

# Gate
if regressions > 2:
    print(f"\n❌ GATE FAILED: Too many regressions ({regressions})")
    exit(1)

if xl_favorable < 5:
    print(f"\n❌ GATE FAILED: XL not maintaining favorable (only {xl_favorable}/6)")
    exit(1)

if overall_delta > 1.0:
    print(f"\n❌ GATE FAILED: Overall worse than baseline by {overall_delta:.2f}%")
    exit(1)

print(f"\n✓ GATE PASSED: Safe to deploy optimization")
exit(0)
PYSCRIPT
```

---

## Stage 4: Documentation & Knowledge Transfer

### Step 4.1: Update Hardware Registry

Create a new entry documenting results on your hardware:

```markdown
# [Your GPU Model] - SIFT GPU Optimization Results

**Date:** [Date]
**Hardware:** [GPU Model, Compute Units, Memory Bandwidth]
**GPU Category:** [iGPU / Entry dGPU / Mid-range dGPU / High-end dGPU]

## Baseline Performance (Phases 1-3 Active)

| Metric | Value |
|--------|-------|
| Favorable Cases | [X]/24 |
| Average GPU/CPU | [X]× |
| vs Ultra 7 255H + Intel Arc Baseline | [+X%] |
| Best Configuration | [config] |

## Optimizations Attempted

- [ ] Phase A (Environment Tuning): [Result]
- [ ] Phase B1 (Orientation Batching): [Result]
- [ ] Phase B2 (Cooperative Kernels): [Result]
- [ ] Phase B3 (Kernel Pragmas): [Skipped]

## Recommendations for Next Agent

[Your insights here]
```

### Step 4.2: Archival

```bash
# Save all results for future reference
mkdir -p /path/to/opencv/optimization_history/$(date +%Y%m%d)
cp logs/* /path/to/opencv/optimization_history/$(date +%Y%m%d)/
cp logs/hw_comparison.md /path/to/opencv/optimization_history/$(date +%Y%m%d)/
```

---

## Troubleshooting Guide

### Problem: Baseline Worse Than Expected

**Symptoms:** New HW baseline 10%+ slower than category expectation

**Diagnosis Steps:**
```bash
# 1. Check OpenCL driver
clinfo | grep "Driver Version"

# 2. Check for thermal throttling
watch -n 1 'clinfo | grep -E "Current|Frequency"'
# Run benchmark in background; watch should show freq staying at max

# 3. Check for context switching
OPENCV_OPENCL_RAISE_ERROR=1 ./build_host/bin/example_tapi_sift_benchmark --warmup=0 --iters=1 --suite
# Should have no ERROR or WARN lines

# 4. Try a different warm-up/iter setting
./build_host/bin/example_tapi_sift_benchmark --warmup=5 --iters=20 --suite
# More iterations reduces noise
```

**Solutions:**
- Update OpenCL driver (may fix 5–10%)
- Disable CPU turbo/governor changes during benchmark
- Check for background processes using GPU

### Problem: Segfault or OpenCL Error During Benchmark

**Diagnosis:**
```bash
export OPENCV_OPENCL_RAISE_ERROR=1
export OPENCV_DEBUG_OPENCL_CONFIG_INTERNAL_CALLS=1
./build_host/bin/example_tapi_sift_benchmark --warmup=0 --iters=1 --suite 2>&1 | head -50
```

**Common Causes:**
- Out of GPU memory (check with `clinfo | grep "Max Memory Alloc"`)
- Incompatible OpenCL compiler version
- Driver crash on certain kernel patterns

**Recovery:**
- Revert last change: `git checkout modules/features2d/src/sift.dispatch.cpp`
- Rebuild: `cmake --build build_host --target opencv_features2d -j$(nproc)`
- Re-test: `./build_host/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite`

### Problem: Phase 5C/5D Makes Performance MUCH Worse

**Expected:** -1 to -3% regression (data shows these regress on Ultra 7 255H + Intel Arc)  
**Observed:** >5% regression

**Reason:** Phase 5C/5D are inherently risky. They added overhead on Ultra 7 255H + Intel Arc; they may add even more overhead on your hardware depending on compiler and architecture.

**Decision:**
- **If HW category is iGPU/Entry dGPU:** Revert immediately. These phases are not suitable.
- **If HW category is Mid-range/High-end dGPU:** Investigate with profiler before reverting.

```bash
# Revert if unsure
git checkout modules/features2d/src/sift.dispatch.cpp
cmake --build build_host --target opencv_features2d -j$(nproc)
```

---

## Reference: Baseline Results by Category

### Ultra 7 255H + Intel Arc (iGPU) — Reference Baseline

| Size | Config | Favorable? | Ratio |
|---|---|---|---:|
| S_small | avg | 0/6 | 1.107× |
| M_medium | avg | 4/6 | 0.970× |
| L_large | avg | 4/6 | 0.943× |
| XL_very_large | avg | 6/6 | 0.920× |
| **Overall** | — | **14/24** | **0.963×** |

### Expected Performance Scaling

| GPU Category | Expected Avg Ratio | Expected Favorable | Notes |
|---|---|---|---|
| Integrated iGPU | 0.960×–0.965× | 13–15/24 | Same as Ultra 7 255H + Intel Arc ±1% |
| Entry dGPU (128–256 EU) | 0.920–0.940× | 16–18/24 | +2–5% faster |
| Mid-range dGPU (256–512 EU) | 0.880–0.920× | 18–20/24 | +5–10% faster |
| High-end dGPU (512+ EU) | 0.800–0.880× | 20–22/24 | +10–20% faster |

**These are guidelines, not guarantees.** Compiler, driver, and architecture details matter significantly.

---

## Key Takeaways for Agents

1. **Start with Stage 0:** Always characterize hardware first; don't assume.

2. **Baseline is the truth:** If new HW baseline differs from expectation, debug before proceeding.

3. **Phase A is safe:** Environment tuning is zero-risk; try it on all hardware categories.

4. **Phase B is risky:** Kernel-level optimizations regressed on Ultra 7 255H + Intel Arc and should be attempted only on clearly faster hardware (discrete GPU category).

5. **Reference the Analysis:** When deciding whether to try an optimization, check SIFT_GPU_OPTIMIZATION_ANALYSIS.md for root causes and why it regressed.

6. **Respect the category:** What works for high-end dGPUs will likely hurt integrated iGPUs. Match strategy to hardware.

7. **Gate early, revert fast:** If optimization doesn't meet gates, revert immediately rather than investigating further.

---

**Document Version:** 1.0 (May 7, 2026)  
**Status:** Ready for autonomous agent execution  
**Reference Analysis:** `SIFT_GPU_OPTIMIZATION_ANALYSIS.md`
