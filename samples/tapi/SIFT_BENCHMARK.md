# SIFT OpenCL benchmark — results and how to reproduce

This document records output from `example_tapi_sift_benchmark` (`sift_benchmark.cpp`) and gives **repeatable** steps for another machine or agent.

---

## What the benchmark measures

For each synthetic image size and each SIFT configuration, it reports **mean milliseconds per** `SIFT::detectAndCompute` over `iters` timed iterations (after `warmup` untimed iterations):

| Column           | Meaning                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **CPU Mat**      | `cv::Mat` input, `cv::ocl::setUseOpenCL(false)`                                                                    |
| **UMat OCL off** | `cv::UMat` input, OpenCL use disabled (CPU path)                                                                   |
| **UMat OCL on**  | `cv::UMat` input, OpenCL enabled (GPU when a GPU is the default OpenCL device and `cv::ocl::haveOpenCL()` is true) |
| **GPU/CPU**      | `UMat OCL on` / `CPU Mat` (values below 1.0 mean the UMat+OCL path was faster than Mat+CPU)                        |

Synthetic images are random 8-bit grayscale with a light Gaussian blur (fixed seed **42**).

---

## Archived results (one captured run)

**Environment (record yours when you update this file):**

| Field             | Value for this archive                                     |
| ----------------- | ---------------------------------------------------------- |
| Host / OS         | Linux (WSL2), x86_64                                       |
| OpenCV            | Local `4.x` build from this tree                           |
| OpenCL at runtime | **Not available** (`OpenCL available: no` from the binary) |
| Benchmark CLI     | `--warmup=0 --iters=1 --suite`                             |
| Date              | 2026-05-03                                                 |

Because OpenCL was unavailable, **UMat OCL on** equals **UMat OCL off** for every row. The table still shows **Mat vs UMat (CPU)** overhead and scaling across sizes and SIFT knobs.

### Results table

| Size          | WxH       | Config      |     Kp | CPU Mat (ms) | UMat OCL off (ms) | UMat OCL on (ms) | GPU/CPU |
| ------------- | --------- | ----------- | -----: | -----------: | ----------------: | ---------------: | ------: |
| S_small       | 512×384   | def_nL3     |   4865 |        25.61 |             15.77 |            15.77 |    0.62 |
| S_small       | 512×384   | layers5     |   7595 |        22.11 |             17.32 |            17.32 |    0.78 |
| S_small       | 512×384   | contrast_hi |    747 |         9.82 |             11.11 |            11.11 |    1.13 |
| S_small       | 512×384   | cap500      |    500 |        10.28 |             10.69 |            10.69 |    1.04 |
| S_small       | 512×384   | edge_tight  |   4759 |        14.86 |             13.58 |            13.58 |    0.91 |
| S_small       | 512×384   | sigma_soft  |   8777 |        13.93 |             14.30 |            14.30 |    1.03 |
| M_medium      | 1280×720  | def_nL3     |  23104 |       109.17 |             86.50 |            86.50 |    0.79 |
| M_medium      | 1280×720  | layers5     |  36036 |       119.30 |             88.28 |            88.28 |    0.74 |
| M_medium      | 1280×720  | contrast_hi |   3541 |        68.01 |             50.74 |            50.74 |    0.75 |
| M_medium      | 1280×720  | cap500      |    501 |        51.53 |             50.41 |            50.41 |    0.98 |
| M_medium      | 1280×720  | edge_tight  |  22435 |        69.73 |             66.41 |            66.41 |    0.95 |
| M_medium      | 1280×720  | sigma_soft  |  41733 |        70.69 |             69.66 |            69.66 |    0.99 |
| L_large       | 1920×1080 | def_nL3     |  52480 |       220.30 |            185.23 |           185.23 |    0.84 |
| L_large       | 1920×1080 | layers5     |  81723 |       251.94 |            203.92 |           203.92 |    0.81 |
| L_large       | 1920×1080 | contrast_hi |   8074 |       180.87 |            114.40 |           114.40 |    0.63 |
| L_large       | 1920×1080 | cap500      |    500 |       151.11 |            114.74 |           114.74 |    0.76 |
| L_large       | 1920×1080 | edge_tight  |  50975 |       186.60 |            156.17 |           156.17 |    0.84 |
| L_large       | 1920×1080 | sigma_soft  |  94478 |       189.97 |            161.43 |           161.43 |    0.85 |
| XL_very_large | 3840×2160 | def_nL3     | 210269 |      1044.88 |            913.71 |           913.71 |    0.87 |
| XL_very_large | 3840×2160 | layers5     | 327239 |      1191.82 |           1130.31 |          1130.31 |    0.95 |
| XL_very_large | 3840×2160 | contrast_hi |  32213 |       653.30 |            642.70 |           642.70 |    0.98 |
| XL_very_large | 3840×2160 | cap500      |    500 |       720.62 |            677.07 |           677.07 |    0.94 |
| XL_very_large | 3840×2160 | edge_tight  | 204502 |      1045.01 |            845.39 |           845.39 |    0.81 |
| XL_very_large | 3840×2160 | sigma_soft  | 377031 |      1060.06 |            878.15 |           878.15 |    0.83 |

### Suite configuration labels

| Tag             | SIFT parameters (high level)                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| **def_nL3**     | Default-like: `nOctaveLayers=3`, `contrastThreshold=0.04`, `edgeThreshold=10`, `sigma=1.6`, uncapped features |
| **layers5**     | More per-octave layers: `nOctaveLayers=5`                                                                     |
| **contrast_hi** | Stricter peak filter: `contrastThreshold=0.12`                                                                |
| **cap500**      | Cap features: `nfeatures=500`                                                                                 |
| **edge_tight**  | Stricter edge rejection: `edgeThreshold=6`                                                                    |
| **sigma_soft**  | Sofper initial blur: `sigma=1.2`                                                                              |

---

## How to reproduce (another system / agent)

### 1. Configure CMake

Minimal flags for the benchmark binary:

- **`BUILD_EXAMPLES=ON`** — required; TAPI samples are skipped otherwise (`samples/tapi/CMakeLists.txt`).
- **`WITH_OPENCL=ON`** — recommended if you want **UMat OCL on** to use a GPU; without OpenCL support in the build, the third column will never diverge from the second at runtime.

Example configure (adjust generator and install prefix as needed):

```bash
cd /path/to/opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DWITH_OPENCL=ON
```

**Note:** Even with `WITH_OPENCL=ON`, `cv::ocl::haveOpenCL()` can still be false until a valid **OpenCL ICD** and driver are installed (e.g. vendor OpenCL for NVIDIA/AMD/Intel). On Linux, `libOpenCL` (often `ocl-icd-libopencl1`) plus the vendor ICD is typical.

### 2. Build the benchmark target

The sample is built as **`example_tapi_sift_benchmark`** (from `sift_benchmark.cpp`).

```bash
cmake --build build --target example_tapi_sift_benchmark -j"$(nproc)"
```

If CMake reports that the target is unknown, confirm `BUILD_EXAMPLES=ON` and that TAPI sample dependencies resolved (core, imgproc, features2d, imgcodecs, video, videoio, highgui, objdetect, calib3d, flann).

### 3. Run with a known library path (Unix install tree)

If OpenCV is not installed system-wide, point the loader at the build’s `lib` directory:

```bash
export LD_LIBRARY_PATH="/path/to/opencv/build/lib:${LD_LIBRARY_PATH}"
./path/to/opencv/build/bin/example_tapi_sift_benchmark --help
```

On **macOS** use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH` (and note SIP restrictions if any).

### 4. Benchmark invocation (repeatable matrix)

**Full synthetic suite** (same matrix as archived above):

```bash
./build/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite
```

Equivalent forms (the sample parses these explicitly):

```bash
./build/bin/example_tapi_sift_benchmark --warmup 2 --iters 8 --suite
```

- With **no arguments**, the program also runs the full suite (same as passing `--suite`).
- **`--help`** prints usage.

**Faster but noisier** (good for CI smoke checks only):

```bash
./build/bin/example_tapi_sift_benchmark --warmup=0 --iters=1 --suite
```

**Capture stdout to a file** (append or replace your results section):

```bash
./build/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite 2>&1 | tee sift_benchmark_run.txt
```

### 5. OpenCL device selection (optional)

When `OpenCL available: yes`, you can steer OpenCV’s default OpenCL device with the usual OpenCV OpenCL environment variables (see OpenCV docs: **OpenCL optimization** / `OPENCV_OPENCL_DEVICE`). Example pattern:

```bash
export OPENCV_OPENCL_DEVICE=:GPU:0
./build/bin/example_tapi_sift_benchmark --warmup=2 --iters=8 --suite
```

Exact syntax depends on OpenCV version and platform; read the header lines printed by the binary (device name and vendor).

### 6. SIFT-specific OpenCL toggles (optional)

Implemented in the SIFT OpenCL path (`sift.dispatch.cpp`):

| Variable                        | Purpose                                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `OPENCV_SIFT_OPENCL_FORCE=1`    | Force attempting OpenCL SIFT on small inputs (default skips very small images to avoid launch overhead). |
| `OPENCV_SIFT_OPENCL_MIN_PIXELS` | Integer; minimum `cols*rows` before OpenCL SIFT is considered (default `196608` = 512×384).              |
| `OPENCV_SIFT_OPENCL_MIN_SIDE`   | Integer; minimum of width/height (default `384`).                                                        |

For apples-to-apples reruns, either leave these unset or document their values in the “Environment” table when you archive a new run.

### 7. Sanity checks before trusting numbers

1. First lines of stdout: **`OpenCL available: yes|no`** and default device line when OpenCL is on.
2. If OpenCL is **no**, treat **GPU/CPU** as “UMat CPU vs Mat CPU” only.
3. Prefer **`--iters` ≥ 5** and **`--warmup` ≥ 2** for stable means unless you only need a quick smoke test.
4. Close other heavy GPU/CPU workloads; pin CPU frequency if you need lab-grade repeatability.

---

## Updating this file after a new run

1. Run the benchmark with documented `cmake` flags, `LD_LIBRARY_PATH`, env vars, and CLI.
2. Copy the printed Markdown table (or full log) from stdout.
3. Replace the **Archived results** section (environment table + results table) and adjust the narrative if OpenCL was available.

---

## Related source

- Benchmark: `samples/tapi/sift_benchmark.cpp` → binary **`example_tapi_sift_benchmark`**
- SIFT OpenCL dispatch and env knobs: `modules/features2d/src/sift.dispatch.cpp`
