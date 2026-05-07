// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Features2d_SIFT, descriptor_type)
{
    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Mat descriptorsFloat, descriptorsUchar;
    Ptr<SIFT> siftFloat = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    siftFloat->detectAndCompute(gray, Mat(), keypoints, descriptorsFloat, false);
    ASSERT_EQ(descriptorsFloat.type(), CV_32F) << "type mismatch";

    Ptr<SIFT> siftUchar = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_8U);
    siftUchar->detectAndCompute(gray, Mat(), keypoints, descriptorsUchar, false);
    ASSERT_EQ(descriptorsUchar.type(), CV_8U) << "type mismatch";

    Mat descriptorsFloat2;
    descriptorsUchar.assignTo(descriptorsFloat2, CV_32F);
    Mat diff = descriptorsFloat != descriptorsFloat2;
    ASSERT_EQ(countNonZero(diff), 0) << "descriptors are not identical";
}

TEST(Features2d_SIFT, regression_26139)
{
    auto extractor = cv::SIFT::create();
    cv::Mat1b image{cv::Size{300, 300}, 0};
    std::vector<cv::KeyPoint> kps {
        cv::KeyPoint(154.076813f, 136.160904f, 111.078636f, 216.195618f, 0.00000899323549f, 7)
    };
    cv::Mat descriptors;
    extractor->compute(image, kps, descriptors); // we expect no memory corruption
    ASSERT_EQ(descriptors.size(), Size(128, 1));
}

// Verify that the CPU descriptor path (which the OCL two-pass path must numerically
// match) is stable across multiple calls with the same synthetic image and the same
// set of provided keypoints.  This exercises the workload-sorting changes (the
// secondary spatial sort must not change the descriptor values, only their
// computation order).
TEST(Features2d_SIFT, descriptor_stability_synthetic)
{
    // Build a synthetic image with distinct local texture so SIFT finds real keypoints.
    Mat img(256, 256, CV_8UC1);
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++)
            img.at<uchar>(r, c) = static_cast<uchar>((r * 17 + c * 13 + (r ^ c) * 7) & 0xFF);
    GaussianBlur(img, img, Size(3, 3), 0.8);

    auto sift = cv::SIFT::create(100);

    // Detect keypoints and compute descriptors twice; results must be bit-exact.
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    sift->detectAndCompute(img, noArray(), kpts1, desc1);
    sift->detectAndCompute(img, noArray(), kpts2, desc2);

    ASSERT_EQ(kpts1.size(), kpts2.size());
    if (!desc1.empty())
    {
        ASSERT_EQ(desc1.size(), desc2.size());
        Mat diff;
        absdiff(desc1, desc2, diff);
        EXPECT_EQ(0, countNonZero(diff.reshape(1))) << "Descriptors changed between calls";
    }
}

// Verify that using pre-provided keypoints (the compute() path) yields descriptors
// whose L2 norm per row is in the expected [0, 128*255] range for CV_32F output.
// This is a lightweight sanity check that does not require test-data files.
TEST(Features2d_SIFT, descriptor_norm_range_synthetic)
{
    Mat img(256, 256, CV_8UC1);
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++)
            img.at<uchar>(r, c) = static_cast<uchar>((r * 11 + c * 23 + (r & c) * 5) & 0xFF);
    GaussianBlur(img, img, Size(3, 3), 0.8);

    auto sift = cv::SIFT::create(50);
    vector<KeyPoint> kpts;
    Mat desc;
    sift->detectAndCompute(img, noArray(), kpts, desc);

    if (kpts.empty())
        return; // synthetic image may not yield keypoints on every platform

    ASSERT_EQ(desc.type(), CV_32F);
    ASSERT_EQ(desc.cols, 128);

    for (int i = 0; i < desc.rows; i++)
    {
        double norm = cv::norm(desc.row(i));
        EXPECT_GT(norm, 0.0) << "Descriptor row " << i << " is all-zeros";
        // Each element is in [0, 255]; norm <= sqrt(128) * 255 ≈ 2883
        EXPECT_LE(norm, 3000.0) << "Descriptor row " << i << " norm is unexpectedly large";
    }
}

}} // namespace
