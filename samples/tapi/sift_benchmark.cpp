/**
 * @file sift_benchmark.cpp
 * @brief Compare SIFT detectAndCompute timing with CPU (Mat) vs T-API (UMat, optional OpenCL).
 *
 * Usage:
 * @code{.sh}
 *   ./example_tapi_sift_benchmark [path/to/image.png]
 * @endcode
 *
 * Use environment variables or cv::ocl::setUseOpenCL to select devices (see OpenCV OpenCL documentation).
 * Run twice: once with OpenCL off for CPU baseline, once with OpenCL on for GPU (if available).
 *
 * Example:
 * @code{.sh}
 *   OPENCV_OPENCL_DEVICE=:CPU:0 ./example_tapi_sift_benchmark ../data/lena.jpg
 *   ./example_tapi_sift_benchmark ../data/lena.jpg
 * @endcode
 */

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;

static void runSift(const String& label, InputArray img, int warmup, int iters)
{
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kpts;
    Mat desc;
    for (int w = 0; w < warmup; w++)
    {
        kpts.clear();
        sift->detectAndCompute(img, noArray(), kpts, desc, false);
    }
    const int64 t0 = getTickCount();
    for (int i = 0; i < iters; i++)
    {
        kpts.clear();
        sift->detectAndCompute(img, noArray(), kpts, desc, false);
    }
    const double sec = (getTickCount() - t0) / getTickFrequency();
    std::cout << label << ": " << iters << " runs in " << sec << " s, "
              << (1000.0 * sec / iters) << " ms/run, keypoints=" << kpts.size()
              << ", desc rows=" << desc.rows << std::endl;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h usage ? | | print this message }"
        "{warmup w       | 1 | warmup iterations }"
        "{iters n        | 5 | timed iterations }"
        "{@image         |   | grayscale or color image }");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String path = parser.get<String>("@image");
    if (path.empty())
    {
        std::cerr << "Please provide an image path.\n";
        parser.printMessage();
        return 1;
    }

    Mat m = imread(path, IMREAD_GRAYSCALE);
    if (m.empty())
    {
        std::cerr << "Cannot read image: " << path << std::endl;
        return 2;
    }

    const int warmup = parser.get<int>("warmup");
    const int iters = std::max(1, parser.get<int>("iters"));

    std::cout << "OpenCL enabled flag: " << (ocl::useOpenCL() ? "true" : "false") << std::endl;
    if (ocl::useOpenCL())
    {
        ocl::Device d = ocl::Device::getDefault();
        std::cout << "OpenCL device: " << d.name() << " (" << d.vendorName() << ")\n";
    }

    runSift("Mat (CPU path)", m, warmup, iters);

    UMat u;
    m.copyTo(u);
    runSift("UMat (OpenCL path when device available)", u, warmup, iters);

    return 0;
}
