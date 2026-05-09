/*
 * End-to-end SIFT + matching + homography benchmark.
 *
 * Modes:
 *   baseline   : BFMatcher(CPU Mat) + RANSAC
 *   match_ocl  : BFMatcher(OpenCL UMat) + RANSAC
 *   match_opt  : FLANN(KDTree) + RANSAC
 *   ransac_opt : BFMatcher(CPU Mat) + RHO
 *   both_ocl   : BFMatcher(OpenCL UMat) + RHO
 *   both       : FLANN(KDTree) + RHO
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <cstring>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/flann.hpp"

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::vector;

enum MatchMode { MATCH_BF_CPU = 0, MATCH_BF_OCL = 1, MATCH_FLANN = 2 };
enum RansacMode { RANSAC_CLASSIC = 0, RANSAC_RHO = 1 };

struct BenchmarkConfig {
    int warmupIters = 2;
    int timedIters = 8;
    float scaleFactor = 1.0f;
    int minMatches = 4;
    float ratioTest = 0.75f;
    double ransacThreshold = 3.0;
    double ransacConfidence = 0.995;
    int ransacMaxIters = 2000;
    int maxMatchDescriptors = 0;  // 0 = no cap (FLANN); BF modes always capped at 65535 (IMGIDX_ONE)
    MatchMode matchMode = MATCH_BF_CPU;
    RansacMode ransacMode = RANSAC_CLASSIC;
};

struct StageStats {
    int kpts1 = 0;
    int kpts2 = 0;
    int rawMatches = 0;
    int goodMatches = 0;
    int inliers = 0;
    double sift1Ms = 0.0;
    double sift2Ms = 0.0;
    double matchMs = 0.0;
    double ransacMs = 0.0;
    double totalMs = 0.0;
    int oclMatchFallbacks = 0;
};

static Mat makeSyntheticGray(int w, int h, uint64_t seed)
{
    Mat m(h, w, CV_8UC1);
    RNG rng(seed ^ ((uint64_t)w << 16) ^ (uint64_t)h);
    rng.fill(m, RNG::UNIFORM, 0, 256);
    GaussianBlur(m, m, Size(5, 5), 0.9, 0.9);

    for (int i = 0; i < (w * h) / 7000; ++i)
    {
        Point c(rng.uniform(0, w), rng.uniform(0, h));
        int r = rng.uniform(4, 20);
        circle(m, c, r, Scalar(rng.uniform(40, 220)), 1, LINE_AA);
    }
    return m;
}

static Mat makeWarped(const Mat& src)
{
    vector<Point2f> from = {
        Point2f(0.f, 0.f), Point2f((float)src.cols, 0.f),
        Point2f(0.f, (float)src.rows), Point2f((float)src.cols, (float)src.rows)
    };
    vector<Point2f> to = {
        Point2f(18.f, 12.f), Point2f((float)src.cols - 26.f, 9.f),
        Point2f(10.f, (float)src.rows - 22.f), Point2f((float)src.cols - 18.f, (float)src.rows - 30.f)
    };
    Mat H = getPerspectiveTransform(from, to);
    Mat warped;
    warpPerspective(src, warped, H, src.size(), INTER_LINEAR, BORDER_REFLECT101);
    return warped;
}

static void ratioFilter(const vector<vector<DMatch> >& knn, float ratio, vector<DMatch>& good)
{
    good.clear();
    for (size_t i = 0; i < knn.size(); ++i)
    {
        if (knn[i].size() >= 2 && knn[i][0].distance < ratio * knn[i][1].distance)
            good.push_back(knn[i][0]);
    }
}

static void capKeypointsAndDescriptors(vector<KeyPoint>& kpts, Mat& desc, int maxDesc)
{
    if (maxDesc <= 0 || (int)kpts.size() <= maxDesc || desc.empty())
        return;
    kpts.resize((size_t)maxDesc);
    desc = desc.rowRange(0, maxDesc).clone();
}

static void capKeypointsAndDescriptors(vector<KeyPoint>& kpts, UMat& desc, int maxDesc)
{
    if (maxDesc <= 0 || (int)kpts.size() <= maxDesc || desc.empty())
        return;
    kpts.resize((size_t)maxDesc);
    desc = desc.rowRange(0, maxDesc);
}

static StageStats runPipeline(const Mat& img1Cpu, const Mat& img2Cpu, const BenchmarkConfig& cfg)
{
    StageStats s;

    const bool useUmat = ocl::haveOpenCL() && ocl::useOpenCL();
    UMat u1, u2;
    if (useUmat)
    {
        img1Cpu.copyTo(u1);
        img2Cpu.copyTo(u2);
    }

    Ptr<SIFT> sift = SIFT::create();
    BFMatcher bf(NORM_L2, false);
    FlannBasedMatcher flann(makePtr<flann::KDTreeIndexParams>(4), makePtr<flann::SearchParams>(32));

    // Warmup
    for (int i = 0; i < cfg.warmupIters; ++i)
    {
        vector<KeyPoint> k;
        UMat d;
        if (useUmat)
            sift->detectAndCompute(u1, noArray(), k, d, false);
        else
        {
            Mat dm;
            sift->detectAndCompute(img1Cpu, noArray(), k, dm, false);
        }
    }

    for (int it = 0; it < cfg.timedIters; ++it)
    {
        TickMeter tm;
        vector<KeyPoint> k1, k2;
        UMat d1u, d2u;
        Mat d1m, d2m;

        tm.start();
        if (useUmat) sift->detectAndCompute(u1, noArray(), k1, d1u, false);
        else sift->detectAndCompute(img1Cpu, noArray(), k1, d1m, false);
        tm.stop();
        s.sift1Ms += tm.getTimeMilli();
        tm.reset();

        tm.start();
        if (useUmat) sift->detectAndCompute(u2, noArray(), k2, d2u, false);
        else sift->detectAndCompute(img2Cpu, noArray(), k2, d2m, false);
        tm.stop();
        s.sift2Ms += tm.getTimeMilli();
        tm.reset();

        // BFMatcher has a hard architectural limit of 65535 descriptors (IMGIDX_ONE).
        // Cap only for BF modes to avoid an assertion crash; FLANN is uncapped.
        if (cfg.matchMode == MATCH_BF_CPU || cfg.matchMode == MATCH_BF_OCL)
        {
            const int bfCap = std::min(cfg.maxMatchDescriptors > 0 ? cfg.maxMatchDescriptors : 65535, 65535);
            if (useUmat)
            {
                capKeypointsAndDescriptors(k1, d1u, bfCap);
                capKeypointsAndDescriptors(k2, d2u, bfCap);
            }
            else
            {
                capKeypointsAndDescriptors(k1, d1m, bfCap);
                capKeypointsAndDescriptors(k2, d2m, bfCap);
            }
        }

        if (it == 0)
        {
            s.kpts1 = (int)k1.size();
            s.kpts2 = (int)k2.size();
        }

        if ((int)k1.size() < cfg.minMatches || (int)k2.size() < cfg.minMatches)
            continue;

        vector<vector<DMatch> > knn;
        tm.start();
        if (cfg.matchMode == MATCH_BF_CPU)
        {
            if (useUmat)
            {
                d1m = d1u.getMat(ACCESS_READ);
                d2m = d2u.getMat(ACCESS_READ);
            }
            bf.knnMatch(d1m, d2m, knn, 2);
        }
        else if (cfg.matchMode == MATCH_BF_OCL)
        {
            // Direct OCL BF path for measurement; fall back to CPU BF if driver rejects UMat handles.
            bool matched = false;
            if (useUmat)
            {
                try
                {
                    bf.knnMatch(d1u, d2u, knn, 2);
                    matched = true;
                }
                catch (const cv::Exception&)
                {
                    matched = false;
                }
            }
            if (!matched)
            {
                if (useUmat)
                {
                    d1m = d1u.getMat(ACCESS_READ);
                    d2m = d2u.getMat(ACCESS_READ);
                }
                bf.knnMatch(d1m, d2m, knn, 2);
                s.oclMatchFallbacks++;
            }
        }
        else
        {
            // FLANN path uses Mat CV_32F descriptors.
            if (useUmat)
            {
                d1m = d1u.getMat(ACCESS_READ);
                d2m = d2u.getMat(ACCESS_READ);
            }
            flann.knnMatch(d1m, d2m, knn, 2);
        }
        tm.stop();
        s.matchMs += tm.getTimeMilli();
        tm.reset();

        vector<DMatch> good;
        ratioFilter(knn, cfg.ratioTest, good);
        s.rawMatches += (int)knn.size();
        s.goodMatches += (int)good.size();

        if ((int)good.size() < cfg.minMatches)
            continue;

        vector<Point2f> p1, p2;
        p1.reserve(good.size());
        p2.reserve(good.size());
        for (size_t i = 0; i < good.size(); ++i)
        {
            p1.push_back(k1[good[i].queryIdx].pt);
            p2.push_back(k2[good[i].trainIdx].pt);
        }

        int method = (cfg.ransacMode == RANSAC_RHO) ? RHO : RANSAC;
        Mat inlierMask;

        tm.start();
        Mat H = findHomography(p1, p2, method, cfg.ransacThreshold, inlierMask, cfg.ransacMaxIters, cfg.ransacConfidence);
        tm.stop();
        s.ransacMs += tm.getTimeMilli();

        if (!H.empty() && !inlierMask.empty())
            s.inliers += countNonZero(inlierMask);
    }

    const double div = (double)cfg.timedIters;
    s.sift1Ms /= div;
    s.sift2Ms /= div;
    s.matchMs /= div;
    s.ransacMs /= div;
    s.totalMs = s.sift1Ms + s.sift2Ms + s.matchMs + s.ransacMs;
    s.rawMatches = (int)(s.rawMatches / div);
    s.goodMatches = (int)(s.goodMatches / div);
    s.inliers = (int)(s.inliers / div);
    return s;
}

static void applyMode(const string& mode, BenchmarkConfig& cfg)
{
    if (mode == "baseline")
    {
        cfg.matchMode = MATCH_BF_CPU;
        cfg.ransacMode = RANSAC_CLASSIC;
    }
    else if (mode == "match_ocl")
    {
        cfg.matchMode = MATCH_BF_OCL;
        cfg.ransacMode = RANSAC_CLASSIC;
    }
    else if (mode == "match_opt")
    {
        cfg.matchMode = MATCH_FLANN;
        cfg.ransacMode = RANSAC_CLASSIC;
    }
    else if (mode == "ransac_opt")
    {
        cfg.matchMode = MATCH_BF_CPU;
        cfg.ransacMode = RANSAC_RHO;
    }
    else if (mode == "both_ocl")
    {
        cfg.matchMode = MATCH_BF_OCL;
        cfg.ransacMode = RANSAC_RHO;
    }
    else if (mode == "both")
    {
        cfg.matchMode = MATCH_FLANN;
        cfg.ransacMode = RANSAC_RHO;
    }
}

int main(int argc, char** argv)
{
    BenchmarkConfig cfg;
    string mode = "baseline";

    for (int i = 1; i < argc; ++i)
    {
        if (strncmp(argv[i], "--warmup=", 9) == 0) cfg.warmupIters = atoi(argv[i] + 9);
        else if (strncmp(argv[i], "--iters=", 8) == 0) cfg.timedIters = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "--scale=", 8) == 0) cfg.scaleFactor = (float)atof(argv[i] + 8);
        else if (strncmp(argv[i], "--mode=", 7) == 0) mode = string(argv[i] + 7);
        else if (strncmp(argv[i], "--max-match-desc=", 17) == 0) cfg.maxMatchDescriptors = atoi(argv[i] + 17);
    }
    applyMode(mode, cfg);

    cout << "========================================================================\n";
    cout << "SIFT + Matching + RANSAC End-to-End Benchmark\n";
    cout << "========================================================================\n";
    cout << "Mode: " << mode << "\n";
    cout << "OpenCL available: " << (ocl::haveOpenCL() ? "yes" : "no") << "\n";
    cout << "OpenCL enabled:   " << (ocl::useOpenCL() ? "yes" : "no") << "\n";
    if (ocl::haveOpenCL())
    {
        ocl::Device dev = ocl::Device::getDefault();
        string type = (dev.type() & ocl::Device::TYPE_GPU) ? "GPU" :
                      ((dev.type() & ocl::Device::TYPE_CPU) ? "CPU" : "Other");
        cout << "OpenCL device:    " << dev.name() << " | " << type << "\n";
    }
    cout << "Warmup=" << cfg.warmupIters << " iters=" << cfg.timedIters
         << " scale=" << cfg.scaleFactor << "\n\n";
    cout << "Max descriptors per image for matching: " << cfg.maxMatchDescriptors << "\n\n";

    vector<std::tuple<string, int, int> > cases;
    cases.push_back(std::make_tuple("Small", 640, 480));
    cases.push_back(std::make_tuple("Medium", 1280, 720));
    cases.push_back(std::make_tuple("Large", 1920, 1080));
    cases.push_back(std::make_tuple("XL", 3840, 2160));

    cout << std::left << std::setw(9) << "Case"
         << std::setw(12) << "Size"
         << std::right << std::setw(8) << "Kp1"
         << std::setw(8) << "Kp2"
         << std::setw(10) << "RawM"
         << std::setw(10) << "GoodM"
            << std::setw(8) << "OCLfb"
         << std::setw(10) << "Inliers"
         << std::setw(12) << "SIFT1"
         << std::setw(12) << "SIFT2"
         << std::setw(12) << "Match"
         << std::setw(12) << "RANSAC"
         << std::setw(12) << "Total"
         << "\n";
        cout << string(135, '-') << "\n";

    for (size_t i = 0; i < cases.size(); ++i)
    {
        int w = (int)(std::get<1>(cases[i]) * cfg.scaleFactor);
        int h = (int)(std::get<2>(cases[i]) * cfg.scaleFactor);
        if (w < 320 || h < 240)
            continue;

        Mat img1 = makeSyntheticGray(w, h, 42 + i);
        Mat img2 = makeWarped(img1);
        StageStats r = runPipeline(img1, img2, cfg);

        cout << std::left << std::setw(9) << std::get<0>(cases[i])
             << (std::to_string(w) + "x" + std::to_string(h));
        cout << std::right
             << std::setw(8) << r.kpts1
             << std::setw(8) << r.kpts2
             << std::setw(10) << r.rawMatches
             << std::setw(10) << r.goodMatches
               << std::setw(8) << r.oclMatchFallbacks
             << std::setw(10) << r.inliers
             << std::fixed << std::setprecision(2)
             << std::setw(12) << r.sift1Ms
             << std::setw(12) << r.sift2Ms
             << std::setw(12) << r.matchMs
             << std::setw(12) << r.ransacMs
             << std::setw(12) << r.totalMs
             << "\n";
    }

    cout << "\nNotes:\n";
    cout << "- Set OPENCV_OPENCL_DEVICE=:GPU:0 and OPENCV_SIFT_OPENCL_FULL=1 for full GPU SIFT path.\n";
    cout << "- Modes: baseline, match_ocl, match_opt, ransac_opt, both_ocl, both\n";
    return 0;
}
