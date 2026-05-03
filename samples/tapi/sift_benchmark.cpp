/**
 * @file sift_benchmark.cpp
 * @brief SIFT CPU vs OpenCL (UMat) benchmark: synthetic sizes, SIFT parameter sweeps, results table.
 *
 * Environment (optional):
 * - OPENCV_SIFT_OPENCL_MIN_PIXELS, OPENCV_SIFT_OPENCL_MIN_SIDE, OPENCV_SIFT_OPENCL_FORCE (see SIFT OpenCL path in OpenCV)
 * - OPENCV_OPENCL_DEVICE / platform selection (OpenCV OpenCL docs)
 *
 * Usage:
 * @code{.sh}
 *   ./example_tapi_sift_benchmark
 *   ./example_tapi_sift_benchmark --suite
 *   ./example_tapi_sift_benchmark --warmup 2 --iters 8 --suite
 * @endcode
 * (With no arguments, the full synthetic suite runs. Warmup/iters accept space or = forms.)
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;

struct SizeSpec {
    const char* tag;
    int w, h;
};

struct SiftCfg {
    const char* tag;
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
};

static Mat makeSynthetic(int w, int h, uint64_t seed)
{
    Mat m(h, w, CV_8UC1);
    RNG rng((uint64_t)(seed ^ ((uint64_t)w << 16) ^ (uint64_t)h));
    rng.fill(m, RNG::UNIFORM, 0, 256);
    GaussianBlur(m, m, Size(5, 5), 0.8, 0.8);
    return m;
}

/** OpenCV CommandLineParser only accepts --key=value (not --key value); parse ints explicitly. */
static bool parseCliInts(int argc, char** argv, int& warmup, int& iters, bool& suite, bool& wantHelp, String& imagePath)
{
    warmup = 2;
    iters = 6;
    suite = false;
    wantHelp = false;
    imagePath.clear();
    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        const String s(a);
        auto eqVal = [&](const char* prefix) -> String {
            const size_t n = strlen(prefix);
            return (s.length() >= n && s.compare(0, n, prefix) == 0) ? s.substr(n) : String();
        };
        if (s == "--help" || s == "-h" || s == "-?" || s == "--usage")
        {
            wantHelp = true;
            continue;
        }
        if (s == "--suite" || s == "-suite")
        {
            suite = true;
            continue;
        }
        String v;
        if ((v = eqVal("--warmup=")).length() || (v = eqVal("-w=")).length())
        {
            warmup = std::max(0, std::atoi(v.c_str()));
            continue;
        }
        if (s == "--warmup" || s == "-w")
        {
            if (i + 1 < argc)
                warmup = std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if ((v = eqVal("--iters=")).length() || (v = eqVal("-n=")).length())
        {
            iters = std::max(1, std::atoi(v.c_str()));
            continue;
        }
        if (s == "--iters" || s == "-n")
        {
            if (i + 1 < argc)
                iters = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (s.length() && s[0] != '-')
        {
            imagePath = s;
            continue;
        }
    }
    if (argc <= 1)
        suite = true;
    return true;
}

static double benchMs(const Ptr<SIFT>& sift, InputArray img, int warmup, int iters,
        std::vector<KeyPoint>& kpts, Mat& desc)
{
    for (int w = 0; w < warmup; ++w)
    {
        kpts.clear();
        sift->detectAndCompute(img, noArray(), kpts, desc, false);
    }
    const int64 t0 = getTickCount();
    for (int i = 0; i < iters; ++i)
    {
        kpts.clear();
        sift->detectAndCompute(img, noArray(), kpts, desc, false);
    }
    const double sec = (getTickCount() - t0) / getTickFrequency();
    return 1000.0 * sec / std::max(1, iters);
}

static void printUsage(const char* app)
{
    std::cout
        << "Usage: " << app << " [options] [image]\n"
        << "  --suite, -suite     Run synthetic size x SIFT-config matrix (default if no image).\n"
        << "  --warmup N, -w N    Warmup iterations (default 2). Also --warmup=N.\n"
        << "  --iters N, -n N     Timed iterations (default 6). Also --iters=N.\n"
        << "  image               Grayscale image path; single Mat vs UMat run (no suite).\n"
        << "Note: OpenCV CommandLineParser-style --flag value is supported here for warmup/iters.\n";
}

int main(int argc, char** argv)
{
    int warmup = 2, iters = 6;
    bool suite = false, wantHelp = false;
    String path;
    parseCliInts(argc, argv, warmup, iters, suite, wantHelp, path);
    if (wantHelp)
    {
        printUsage(argv[0]);
        return 0;
    }
    iters = std::max(1, iters);

    std::cout << "OpenCL available: " << (ocl::haveOpenCL() ? "yes" : "no") << "\n";
    std::cout << "OpenCL use flag (initial): " << (ocl::useOpenCL() ? "true" : "false") << "\n";
    if (ocl::haveOpenCL() && ocl::useOpenCL())
    {
        ocl::Device d = ocl::Device::getDefault();
        std::cout << "OpenCL default device: " << d.name() << " | vendor: " << d.vendorName()
                  << " | type: " << (d.type() == ocl::Device::TYPE_GPU ? "GPU" : "CPU/other") << "\n";
    }
    std::cout << std::endl;

    if (!suite && !path.empty())
    {
        Mat m = imread(path, IMREAD_GRAYSCALE);
        if (m.empty())
        {
            std::cerr << "Cannot read: " << path << std::endl;
            return 2;
        }
        Ptr<SIFT> sift = SIFT::create();
        std::vector<KeyPoint> k0, k1;
        Mat d0, d1;
        ocl::setUseOpenCL(false);
        double t0 = benchMs(sift, m, warmup, iters, k0, d0);
        ocl::setUseOpenCL(true);
        UMat u;
        m.copyTo(u);
        double t1 = benchMs(sift, u, warmup, iters, k1, d1);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Mat CPU-only avg: " << t0 << " ms | UMat+OpenCL avg: " << t1 << " ms\n";
        return 0;
    }

    if (!suite)
    {
        std::cerr << "Use --suite for the benchmark matrix, or pass an image path.\n";
        printUsage(argv[0]);
        return 1;
    }

    static const SizeSpec sizes[] = {
        { "S_small", 512, 384 },
        { "M_medium", 1280, 720 },
        { "L_large", 1920, 1080 },
        { "XL_very_large", 3840, 2160 },
    };

    static const SiftCfg cfgs[] = {
        { "def_nL3", 0, 3, 0.04, 10.0, 1.6 },
        { "layers5", 0, 5, 0.04, 10.0, 1.6 },
        { "contrast_hi", 0, 3, 0.12, 10.0, 1.6 },
        { "cap500", 500, 3, 0.04, 10.0, 1.6 },
        { "edge_tight", 0, 3, 0.04, 6.0, 1.6 },
        { "sigma_soft", 0, 3, 0.04, 10.0, 1.2 },
    };

    std::cout << "### SIFT benchmark (ms per detectAndCompute, lower is better)\n";
    std::cout << "Warmup=" << warmup << " iters=" << iters << "\n\n";

    std::cout << "| Size | WxH | Config | Kp | CPU Mat | UMat OCL off | UMat OCL on | GPU/CPU |\n";
    std::cout << "|---|---:|---|---:|---:|---:|---:|---:|\n";

    for (const SizeSpec& sz : sizes)
    {
        Mat m = makeSynthetic(sz.w, sz.h, 42);
        UMat u;
        m.copyTo(u);

        for (const SiftCfg& cfg : cfgs)
        {
            Ptr<SIFT> sift = SIFT::create(cfg.nfeatures, cfg.nOctaveLayers,
                cfg.contrastThreshold, cfg.edgeThreshold, cfg.sigma, CV_32F, true);

            std::vector<KeyPoint> kcpu, kum0, kum1;
            Mat dcpu, dum0, dum1;

            ocl::setUseOpenCL(false);
            double ms_cpu = benchMs(sift, m, warmup, iters, kcpu, dcpu);

            ocl::setUseOpenCL(false);
            double ms_um_off = benchMs(sift, u, warmup, iters, kum0, dum0);

            double ms_um_on = ms_um_off;
            size_t kp_on = kum0.size();
            if (ocl::haveOpenCL())
            {
                ocl::setUseOpenCL(true);
                ms_um_on = benchMs(sift, u, warmup, iters, kum1, dum1);
                kp_on = kum1.size();
            }

            const double ratio = (ms_cpu > 1e-6) ? (ms_um_on / ms_cpu) : 0.0;

            std::cout << "| " << sz.tag << " | " << sz.w << "x" << sz.h << " | " << cfg.tag
                      << " | " << kp_on << " | " << std::fixed << std::setprecision(2) << ms_cpu << " | "
                      << ms_um_off << " | " << ms_um_on << " | " << std::setprecision(2) << ratio << " |\n";
        }
    }

    std::cout << "\nNotes:\n"
              << "- **CPU Mat**: `cv::Mat` input, OpenCL disabled.\n"
              << "- **UMat OCL off**: `cv::UMat` input but `cv::ocl::setUseOpenCL(false)` (CPU path).\n"
              << "- **UMat OCL on**: `cv::UMat` with OpenCL enabled (GPU when default device is GPU).\n"
              << "- **GPU/CPU**: ratio `UMat OCL on` / `CPU Mat` (<1 means GPU faster).\n"
              << "- OpenCL SIFT may auto-skip on tiny images unless OPENCV_SIFT_OPENCL_FORCE=1.\n";

    return 0;
}
