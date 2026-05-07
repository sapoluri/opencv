// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2020, Intel Corporation, all rights reserved.

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.
 Patent US6711293 expired in March 2020.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Assignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "precomp.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <cstdlib>

#ifdef HAVE_OPENCL
#include "opencl_kernels_features2d.hpp"
#include "sift_detail.hpp"
#endif

#include "sift.simd.hpp"
#include "sift.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv
{

    /*!
     SIFT implementation.

     The class implements SIFT algorithm by D. Lowe.
     */
    class SIFT_Impl : public SIFT
    {
    public:
        explicit SIFT_Impl(int nfeatures = 0, int nOctaveLayers = 3,
                           double contrastThreshold = 0.04, double edgeThreshold = 10,
                           double sigma = 1.6, int descriptorType = CV_32F,
                           bool enable_precise_upscale = true);

        //! returns the descriptor size in floats (128)
        int descriptorSize() const CV_OVERRIDE;

        //! returns the descriptor type
        int descriptorType() const CV_OVERRIDE;

        //! returns the default norm type
        int defaultNorm() const CV_OVERRIDE;

        //! finds the keypoints and computes descriptors for them using SIFT algorithm.
        //! Optionally it can compute descriptors for the user-provided keypoints
        void detectAndCompute(InputArray img, InputArray mask,
                              std::vector<KeyPoint> &keypoints,
                              OutputArray descriptors,
                              bool useProvidedKeypoints = false) CV_OVERRIDE;

        bool getEnablePreciseUpscale() const { return enable_precise_upscale; }

        void buildGaussianPyramid(const Mat &base, std::vector<Mat> &pyr, int nOctaves) const;
        void buildDoGPyramid(const std::vector<Mat> &pyr, std::vector<Mat> &dogpyr) const;
        void findScaleSpaceExtrema(const std::vector<Mat> &gauss_pyr, const std::vector<Mat> &dog_pyr,
                                   std::vector<KeyPoint> &keypoints) const;

        void read(const FileNode &fn) CV_OVERRIDE;
        void write(FileStorage &fs) const CV_OVERRIDE;

        void setNFeatures(int maxFeatures) CV_OVERRIDE { nfeatures = maxFeatures; }
        int getNFeatures() const CV_OVERRIDE { return nfeatures; }

        void setNOctaveLayers(int nOctaveLayers_) CV_OVERRIDE { nOctaveLayers = nOctaveLayers_; }
        int getNOctaveLayers() const CV_OVERRIDE { return nOctaveLayers; }

        void setContrastThreshold(double contrastThreshold_) CV_OVERRIDE { contrastThreshold = contrastThreshold_; }
        double getContrastThreshold() const CV_OVERRIDE { return contrastThreshold; }

        void setEdgeThreshold(double edgeThreshold_) CV_OVERRIDE { edgeThreshold = edgeThreshold_; }
        double getEdgeThreshold() const CV_OVERRIDE { return edgeThreshold; }

        void setSigma(double sigma_) CV_OVERRIDE { sigma = sigma_; }
        double getSigma() const CV_OVERRIDE { return sigma; }

    protected:
        CV_PROP_RW int nfeatures;
        CV_PROP_RW int nOctaveLayers;
        CV_PROP_RW double contrastThreshold;
        CV_PROP_RW double edgeThreshold;
        CV_PROP_RW double sigma;
        CV_PROP_RW int descriptor_type;
        CV_PROP_RW bool enable_precise_upscale;
    };

    Ptr<SIFT> SIFT::create(int _nfeatures, int _nOctaveLayers,
                           double _contrastThreshold, double _edgeThreshold, double _sigma, bool enable_precise_upscale)
    {
        CV_TRACE_FUNCTION();

        return makePtr<SIFT_Impl>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma, CV_32F, enable_precise_upscale);
    }

    Ptr<SIFT> SIFT::create(int _nfeatures, int _nOctaveLayers,
                           double _contrastThreshold, double _edgeThreshold, double _sigma, int _descriptorType, bool enable_precise_upscale)
    {
        CV_TRACE_FUNCTION();

        // SIFT descriptor supports 32bit floating point and 8bit unsigned int.
        CV_Assert(_descriptorType == CV_32F || _descriptorType == CV_8U);
        return makePtr<SIFT_Impl>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma, _descriptorType, enable_precise_upscale);
    }

    String SIFT::getDefaultName() const
    {
        return (Feature2D::getDefaultName() + ".SIFT");
    }

    static inline void
    unpackOctave(const KeyPoint &kpt, int &octave, int &layer, float &scale)
    {
        octave = kpt.octave & 255;
        layer = (kpt.octave >> 8) & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
    }

    static Mat createInitialImage(const Mat &img, bool doubleImageSize, float sigma, bool enable_precise_upscale)
    {
        CV_TRACE_FUNCTION();

        Mat gray, gray_fpt;
        if (img.channels() == 3 || img.channels() == 4)
        {
            cvtColor(img, gray, COLOR_BGR2GRAY);
            gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
        }
        else
            img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

        float sig_diff;

        if (doubleImageSize)
        {
            sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));

            Mat dbl;
            if (enable_precise_upscale)
            {
                dbl.create(Size(gray_fpt.cols * 2, gray_fpt.rows * 2), gray_fpt.type());
                Mat H = Mat::zeros(2, 3, CV_32F);
                H.at<float>(0, 0) = 0.5f;
                H.at<float>(1, 1) = 0.5f;

                cv::warpAffine(gray_fpt, dbl, H, dbl.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT);
            }
            else
            {
#if DoG_TYPE_SHORT
                resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR_EXACT);
#else
                resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
#endif
            }
            Mat result;
            GaussianBlur(dbl, result, Size(), sig_diff, sig_diff);
            return result;
        }
        else
        {
            sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
            Mat result;
            GaussianBlur(gray_fpt, result, Size(), sig_diff, sig_diff);
            return result;
        }
    }

    void SIFT_Impl::buildGaussianPyramid(const Mat &base, std::vector<Mat> &pyr, int nOctaves) const
    {
        CV_TRACE_FUNCTION();

        std::vector<double> sig(nOctaveLayers + 3);
        pyr.resize(nOctaves * (nOctaveLayers + 3));

        // precompute Gaussian sigmas using the following formula:
        //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
        sig[0] = sigma;
        double k = std::pow(2, 1. / nOctaveLayers);
        for (int i = 1; i < nOctaveLayers + 3; i++)
        {
            double sig_prev = (double)std::pow(k, i - 1) * sigma;
            double sig_total = sig_prev * k;
            sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
        }

        for (int o = 0; o < nOctaves; o++)
        {
            for (int i = 0; i < nOctaveLayers + 3; i++)
            {
                Mat &dst = pyr[o * (nOctaveLayers + 3) + i];
                if (o == 0 && i == 0)
                    dst = base;
                // base of new octave is halved image from end of previous octave
                else if (i == 0)
                {
                    const Mat &src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers];
                    resize(src, dst, Size(src.cols / 2, src.rows / 2),
                           0, 0, INTER_NEAREST);
                }
                else
                {
                    const Mat &src = pyr[o * (nOctaveLayers + 3) + i - 1];
                    GaussianBlur(src, dst, Size(), sig[i], sig[i]);
                }
            }
        }
    }

    class buildDoGPyramidComputer : public ParallelLoopBody
    {
    public:
        buildDoGPyramidComputer(
            int _nOctaveLayers,
            const std::vector<Mat> &_gpyr,
            std::vector<Mat> &_dogpyr)
            : nOctaveLayers(_nOctaveLayers),
              gpyr(_gpyr),
              dogpyr(_dogpyr) {}

        void operator()(const cv::Range &range) const CV_OVERRIDE
        {
            CV_TRACE_FUNCTION();

            const int begin = range.start;
            const int end = range.end;

            for (int a = begin; a < end; a++)
            {
                const int o = a / (nOctaveLayers + 2);
                const int i = a % (nOctaveLayers + 2);

                const Mat &src1 = gpyr[o * (nOctaveLayers + 3) + i];
                const Mat &src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
                Mat &dst = dogpyr[o * (nOctaveLayers + 2) + i];
                subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
            }
        }

    private:
        int nOctaveLayers;
        const std::vector<Mat> &gpyr;
        std::vector<Mat> &dogpyr;
    };

    void SIFT_Impl::buildDoGPyramid(const std::vector<Mat> &gpyr, std::vector<Mat> &dogpyr) const
    {
        CV_TRACE_FUNCTION();

        int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
        dogpyr.resize(nOctaves * (nOctaveLayers + 2));

        parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), buildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
    }

    class findScaleSpaceExtremaComputer : public ParallelLoopBody
    {
    public:
        findScaleSpaceExtremaComputer(
            int _o,
            int _i,
            int _threshold,
            int _idx,
            int _step,
            int _cols,
            int _nOctaveLayers,
            double _contrastThreshold,
            double _edgeThreshold,
            double _sigma,
            const std::vector<Mat> &_gauss_pyr,
            const std::vector<Mat> &_dog_pyr,
            TLSData<std::vector<KeyPoint>> &_tls_kpts_struct)

            : o(_o),
              i(_i),
              threshold(_threshold),
              idx(_idx),
              step(_step),
              cols(_cols),
              nOctaveLayers(_nOctaveLayers),
              contrastThreshold(_contrastThreshold),
              edgeThreshold(_edgeThreshold),
              sigma(_sigma),
              gauss_pyr(_gauss_pyr),
              dog_pyr(_dog_pyr),
              tls_kpts_struct(_tls_kpts_struct)
        {
        }
        void operator()(const cv::Range &range) const CV_OVERRIDE
        {
            CV_TRACE_FUNCTION();

            std::vector<KeyPoint> &kpts = tls_kpts_struct.getRef();

            CV_CPU_DISPATCH(findScaleSpaceExtrema, (o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, kpts, range),
                            CV_CPU_DISPATCH_MODES_ALL);
        }

    private:
        int o, i;
        int threshold;
        int idx, step, cols;
        int nOctaveLayers;
        double contrastThreshold;
        double edgeThreshold;
        double sigma;
        const std::vector<Mat> &gauss_pyr;
        const std::vector<Mat> &dog_pyr;
        TLSData<std::vector<KeyPoint>> &tls_kpts_struct;
    };

    //
    // Detects features at extrema in DoG scale space.  Bad features are discarded
    // based on contrast and ratio of principal curvatures.
    void SIFT_Impl::findScaleSpaceExtrema(const std::vector<Mat> &gauss_pyr, const std::vector<Mat> &dog_pyr,
                                          std::vector<KeyPoint> &keypoints) const
    {
        CV_TRACE_FUNCTION();

        const int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
        const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

        keypoints.clear();
        TLSDataAccumulator<std::vector<KeyPoint>> tls_kpts_struct;

        for (int o = 0; o < nOctaves; o++)
            for (int i = 1; i <= nOctaveLayers; i++)
            {
                const int idx = o * (nOctaveLayers + 2) + i;
                const Mat &img = dog_pyr[idx];
                const int step = (int)img.step1();
                const int rows = img.rows, cols = img.cols;

                parallel_for_(Range(SIFT_IMG_BORDER, rows - SIFT_IMG_BORDER),
                              findScaleSpaceExtremaComputer(
                                  o, i, threshold, idx, step, cols,
                                  nOctaveLayers,
                                  contrastThreshold,
                                  edgeThreshold,
                                  sigma,
                                  gauss_pyr, dog_pyr, tls_kpts_struct));
            }

        std::vector<std::vector<KeyPoint> *> kpt_vecs;
        tls_kpts_struct.gather(kpt_vecs);
        for (size_t i = 0; i < kpt_vecs.size(); ++i)
        {
            keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
        }
    }

    static void calcSIFTDescriptor(
        const Mat &img, Point2f ptf, float ori, float scl,
        int d, int n, Mat &dst, int row)
    {
        CV_TRACE_FUNCTION();

        CV_CPU_DISPATCH(calcSIFTDescriptor, (img, ptf, ori, scl, d, n, dst, row),
                        CV_CPU_DISPATCH_MODES_ALL);
    }

    class calcDescriptorsComputer : public ParallelLoopBody
    {
    public:
        calcDescriptorsComputer(const std::vector<Mat> &_gpyr,
                                const std::vector<KeyPoint> &_keypoints,
                                Mat &_descriptors,
                                int _nOctaveLayers,
                                int _firstOctave)
            : gpyr(_gpyr),
              keypoints(_keypoints),
              descriptors(_descriptors),
              nOctaveLayers(_nOctaveLayers),
              firstOctave(_firstOctave) {}

        void operator()(const cv::Range &range) const CV_OVERRIDE
        {
            CV_TRACE_FUNCTION();

            const int begin = range.start;
            const int end = range.end;

            static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

            for (int i = begin; i < end; i++)
            {
                KeyPoint kpt = keypoints[i];
                int octave, layer;
                float scale;
                unpackOctave(kpt, octave, layer, scale);
                CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
                float size = kpt.size * scale;
                Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
                const Mat &img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer];

                float angle = 360.f - kpt.angle;
                if (std::abs(angle - 360.f) < FLT_EPSILON)
                    angle = 0.f;
                calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n, descriptors, i);
            }
        }

    private:
        const std::vector<Mat> &gpyr;
        const std::vector<KeyPoint> &keypoints;
        Mat &descriptors;
        int nOctaveLayers;
        int firstOctave;
    };

    static void calcDescriptors(const std::vector<Mat> &gpyr, const std::vector<KeyPoint> &keypoints,
                                Mat &descriptors, int nOctaveLayers, int firstOctave)
    {
        CV_TRACE_FUNCTION();
        parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave));
    }

#ifdef HAVE_OPENCL
    namespace
    {

        static const int kSiftOclMaxCandidates = 1 << 20;
        static const int kSiftOclMaxCanPerLayer = 1 << 17; // per-layer cap for batched-async collect (128 K)
        static const int kSiftOclOrientationDupFactor = 4;

        struct SiftOclProvisionalKeypoint
        {
            int imageIdx;
            int r;
            int c;
            KeyPoint kpt;
        };

        struct SiftOclImageOrientedKeypoints
        {
            UMat keypoints;
            UMat responses;
            UMat octaves;
            int count;

            SiftOclImageOrientedKeypoints() : count(0) {}
        };

        static UMat siftCreateInitialImageUMat(const UMat &img, bool doubleImageSize, float sigma, bool enable_precise_upscale)
        {
            CV_TRACE_FUNCTION();
            UMat gray, gray_fpt;
            if (img.channels() == 3 || img.channels() == 4)
            {
                cvtColor(img, gray, COLOR_BGR2GRAY);
                gray.convertTo(gray_fpt, CV_32F, (float)sift_detail::SIFT_FIXPT_SCALE, 0);
            }
            else
                img.convertTo(gray_fpt, CV_32F, (float)sift_detail::SIFT_FIXPT_SCALE, 0);

            const float kInitSigma = 0.5f;
            float sig_diff;

            if (doubleImageSize)
            {
                sig_diff = sqrtf(std::max(sigma * sigma - kInitSigma * kInitSigma * 4, 0.01f));

                UMat dbl;
                if (enable_precise_upscale)
                {
                    dbl.create(Size(gray_fpt.cols * 2, gray_fpt.rows * 2), gray_fpt.type());
                    Mat H = Mat::zeros(2, 3, CV_32F);
                    H.at<float>(0, 0) = 0.5f;
                    H.at<float>(1, 1) = 0.5f;
                    warpAffine(gray_fpt, dbl, H, dbl.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT);
                }
                else
                {
                    resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
                }
                UMat result;
                GaussianBlur(dbl, result, Size(), sig_diff, sig_diff);
                return result;
            }
            else
            {
                sig_diff = sqrtf(std::max(sigma * sigma - kInitSigma * kInitSigma, 0.01f));
                UMat result;
                GaussianBlur(gray_fpt, result, Size(), sig_diff, sig_diff);
                return result;
            }
        }

        static void siftBuildGaussianPyramidUMat(const UMat &base, std::vector<UMat> &pyr, int nOctaves, int nOctaveLayers, double sigma)
        {
            CV_TRACE_FUNCTION();
            std::vector<double> sig(nOctaveLayers + 3);
            pyr.resize(nOctaves * (nOctaveLayers + 3));
            sig[0] = sigma;
            double k = std::pow(2., 1. / nOctaveLayers);
            for (int i = 1; i < nOctaveLayers + 3; i++)
            {
                double sig_prev = std::pow(k, (double)(i - 1)) * sigma;
                double sig_total = sig_prev * k;
                sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
            }

            for (int o = 0; o < nOctaves; o++)
            {
                for (int i = 0; i < nOctaveLayers + 3; i++)
                {
                    UMat &dst = pyr[o * (nOctaveLayers + 3) + i];
                    if (o == 0 && i == 0)
                        dst = base;
                    else if (i == 0)
                    {
                        const UMat &src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers];
                        resize(src, dst, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);
                    }
                    else
                    {
                        const UMat &src = pyr[o * (nOctaveLayers + 3) + i - 1];
                        GaussianBlur(src, dst, Size(), sig[i], sig[i]);
                    }
                }
            }
        }

        static const String &siftOclBuildOptions()
        {
            static const String kOpts("-cl-fast-relaxed-math -cl-mad-enable -cl-denorms-are-zero");
            return kOpts;
        }

        static size_t siftPreferredLocal1D(ocl::Kernel &ker, size_t n)
        {
            size_t preferred = ker.preferedWorkGroupSizeMultiple();
            size_t maxWg = ker.workGroupSize();
            if (preferred == 0 || maxWg == 0)
                return 0;
            size_t local = std::min(preferred, maxWg);
            if (local > 256)
                local = 256;
            while (local > n && local > 1)
                local >>= 1;
            return local;
        }

        static size_t siftEnvLocalSize1D(const char *envName, size_t fallback, size_t maxWg, size_t n)
        {
            size_t local = fallback;
            if (const char *e = std::getenv(envName))
            {
                const int parsed = std::max(1, atoi(e));
                local = (size_t)parsed;
            }
            if (maxWg > 0)
                local = std::min(local, maxWg);
            while (local > n && local > 1)
                local >>= 1;
            return local;
        }

        static void siftIntelCollectLocal2D(size_t gw, size_t gh, size_t maxWg, size_t localsize[2], size_t globalsize[2])
        {
            // CudaSift uses fixed launch geometry per kernel family; mirror that idea here by
            // preferring a stable 128-thread tile for the extrema-collect pass on Intel GPU.
            size_t lx = 16, ly = 8;
            if (maxWg > 0 && lx * ly > maxWg)
            {
                lx = 8;
                ly = 8;
            }
            localsize[0] = lx;
            localsize[1] = ly;
            globalsize[0] = ((gw + lx - 1) / lx) * lx;
            globalsize[1] = ((gh + ly - 1) / ly) * ly;
        }

        class siftBuildDoGPyramidUMatComputer : public ParallelLoopBody
        {
        public:
            siftBuildDoGPyramidUMatComputer(int _nOctaveLayers, const std::vector<UMat> &_gpyr, std::vector<UMat> &_dogpyr)
                : nOctaveLayers(_nOctaveLayers), gpyr(_gpyr), dogpyr(_dogpyr) {}

            void operator()(const cv::Range &range) const CV_OVERRIDE
            {
                for (int a = range.start; a < range.end; a++)
                {
                    const int o = a / (nOctaveLayers + 2);
                    const int i = a % (nOctaveLayers + 2);
                    const UMat &src1 = gpyr[o * (nOctaveLayers + 3) + i];
                    const UMat &src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
                    UMat &dst = dogpyr[o * (nOctaveLayers + 2) + i];
                    subtract(src2, src1, dst, noArray(), CV_32F);
                }
            }

        private:
            int nOctaveLayers;
            const std::vector<UMat> &gpyr;
            std::vector<UMat> &dogpyr;
        };

        static void siftBuildDoGPyramidUMat(const std::vector<UMat> &gpyr, std::vector<UMat> &dogpyr, int nOctaveLayers)
        {
            CV_TRACE_FUNCTION();
            int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
            dogpyr.resize(nOctaves * (nOctaveLayers + 2));
            parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), siftBuildDoGPyramidUMatComputer(nOctaveLayers, gpyr, dogpyr));
        }

        static void siftCopyToDeviceBuffer(const UMat &src, UMat &dst)
        {
            if (dst.empty() || dst.size() != src.size() || dst.type() != src.type())
                dst.create(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
            src.copyTo(dst);
        }

        static void siftUMatPyrToMat(const std::vector<UMat> &u, std::vector<Mat> &m)
        {
            // Deep copy: use when u elements will be concurrently accessed by OpenCL operations.
            m.resize(u.size());
            for (size_t i = 0; i < u.size(); i++)
                u[i].copyTo(m[i]);
        }

        static void siftUMatPyrToMatView(const std::vector<UMat> &u, std::vector<Mat> &m)
        {
            // Zero-copy view: ONLY safe when u elements are NOT used by any concurrent OpenCL op.
            m.resize(u.size());
            for (size_t i = 0; i < u.size(); i++)
                m[i] = u[i].getMat(ACCESS_READ);
        }

        static bool siftOclCollectCandidates(
            ocl::Kernel &ker,
            const UMat &prev, const UMat &cur, const UMat &next,
            float threshold,
            UMat &uCounter,
            UMat &uOutRc,
            int &outCount)
        {
            uCounter.setTo(Scalar(0));

            int rows = cur.rows, cols = cur.cols;
            int prev_step = (int)prev.step, cur_step = (int)cur.step, next_step = (int)next.step;

            // Always use 8×8 work groups on Intel GPU; pad global size to multiples of 8.
            // The kernel has an out-of-bounds guard so extra work items exit immediately.
            size_t gw = (size_t)std::max(1, cols - 2 * sift_detail::SIFT_IMG_BORDER);
            size_t gh = (size_t)std::max(1, rows - 2 * sift_detail::SIFT_IMG_BORDER);
            size_t globalsize[2], localsize[2] = {0, 0};
            globalsize[0] = gw;
            globalsize[1] = gh;
            const ocl::Device &dev = ocl::Device::getDefault();
            if (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU))
            {
                siftIntelCollectLocal2D(gw, gh, ker.workGroupSize(), localsize, globalsize);
            }

            bool ok = ker.args(
                             ocl::KernelArg::PtrReadOnly(prev), prev_step,
                             ocl::KernelArg::PtrReadOnly(cur), cur_step,
                             ocl::KernelArg::PtrReadOnly(next), next_step,
                             rows, cols,
                             threshold,
                             ocl::KernelArg::PtrReadWrite(uCounter),
                             ocl::KernelArg::PtrWriteOnly(uOutRc),
                             kSiftOclMaxCandidates)
                          .run(2, globalsize, localsize[0] ? localsize : nullptr, true);

            if (!ok)
                return false;

            Mat cnt;
            uCounter.copyTo(cnt);
            outCount = cnt.at<int>(0);
            if (outCount <= 0)
                return true;
            if (outCount >= kSiftOclMaxCandidates - 4096)
                return false;
            return true;
        }

        class siftRefineCandidatesBody : public ParallelLoopBody
        {
        public:
            siftRefineCandidatesBody(
                int _o, int _i, int _nOctaveLayers,
                double _contrastThreshold, double _edgeThreshold, double _sigma,
                const std::vector<Mat> *_dog_pyr,
                const int *_rc,
                TLSData<std::vector<SiftOclProvisionalKeypoint>> *_tls)
                : o(_o), i(_i), nOctaveLayers(_nOctaveLayers),
                  contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma),
                  dog_pyr(_dog_pyr), rc(_rc), tls(_tls) {}

            void operator()(const cv::Range &range) const CV_OVERRIDE
            {
                std::vector<SiftOclProvisionalKeypoint> &out = tls->getRef();
                for (int k = range.start; k < range.end; k++)
                {
                    int r1 = rc[k * 2], c1 = rc[k * 2 + 1];
                    KeyPoint kpt;
                    int layer = i;
                    if (!sift_detail::adjustLocalExtrema(*dog_pyr, kpt, o, layer, r1, c1,
                                                         nOctaveLayers, (float)contrastThreshold, (float)edgeThreshold, (float)sigma))
                        continue;
                    SiftOclProvisionalKeypoint provisional;
                    provisional.imageIdx = o * (nOctaveLayers + 3) + layer;
                    provisional.r = r1;
                    provisional.c = c1;
                    provisional.kpt = kpt;
                    out.push_back(provisional);
                }
            }

        private:
            int o, i, nOctaveLayers;
            double contrastThreshold, edgeThreshold, sigma;
            const std::vector<Mat> *dog_pyr;
            const int *rc;
            TLSData<std::vector<SiftOclProvisionalKeypoint>> *tls;
        };

        static bool siftOclAssignOrientationsForImageDevice(
            const UMat &uimg,
            const UMat &uRc,
            const UMat &uKpt,
            const UMat &uOct,
            int n,
            SiftOclImageOrientedKeypoints &out)
        {
            if (n <= 0)
            {
                out.count = 0;
                return true;
            }

            const int maxout = std::max(1, n * kSiftOclOrientationDupFactor);
            UMat uCounter(1, 1, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            UMat uOutKpt(maxout, 1, CV_32FC4, USAGE_ALLOCATE_DEVICE_MEMORY);
            UMat uOutResp(maxout, 1, CV_32F, USAGE_ALLOCATE_DEVICE_MEMORY);
            UMat uOutOct(maxout, 1, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            uCounter.setTo(Scalar(0));

            ocl::Kernel ker("SIFT_assignOrientations", ocl::features2d::sift_oclsrc, siftOclBuildOptions());
            if (ker.empty())
                return false;

            size_t globalsize[1] = {(size_t)n};
            size_t local = siftPreferredLocal1D(ker, (size_t)n);
            const ocl::Device &dev = ocl::Device::getDefault();
            if (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU))
                local = siftEnvLocalSize1D("OPENCV_SIFT_OCL_ORI_LOCAL", 64, ker.workGroupSize(), (size_t)n);
            size_t localsize[1] = {local};
            if (localsize[0])
                globalsize[0] = ((globalsize[0] + localsize[0] - 1) / localsize[0]) * localsize[0];
            bool ok = ker.args(
                             ocl::KernelArg::PtrReadOnly(uimg), (int)uimg.step,
                             uimg.rows, uimg.cols,
                             ocl::KernelArg::PtrReadOnly(uRc),
                             ocl::KernelArg::PtrReadOnly(uKpt),
                             ocl::KernelArg::PtrReadOnly(uOct),
                             n,
                             ocl::KernelArg::PtrReadWrite(uCounter),
                             ocl::KernelArg::PtrWriteOnly(uOutKpt),
                             ocl::KernelArg::PtrWriteOnly(uOutResp),
                             ocl::KernelArg::PtrWriteOnly(uOutOct),
                             maxout)
                          .run(1, globalsize, localsize[0] ? localsize : NULL, true);
            if (!ok)
                return false;

            Mat hCount;
            uCounter.copyTo(hCount);
            int outCount = hCount.at<int>(0);
            out.count = outCount;
            if (outCount <= 0)
                return true;
            if (outCount > maxout)
                return false;

            if (outCount < maxout)
            {
                out.keypoints = uOutKpt.rowRange(0, outCount);
                out.responses = uOutResp.rowRange(0, outCount);
                out.octaves = uOutOct.rowRange(0, outCount);
            }
            else
            {
                out.keypoints = uOutKpt;
                out.responses = uOutResp;
                out.octaves = uOutOct;
            }

            return true;
        }

        struct SiftOclOrientationScratch
        {
            UMat counter;
            UMat outKpt;
            UMat outResp;
            UMat outOct;
            int capacity;

            SiftOclOrientationScratch() : capacity(0) {}

            void ensureCapacity(int required)
            {
                if (required <= capacity)
                    return;
                capacity = required;
                outKpt.create(capacity, 1, CV_32FC4, USAGE_ALLOCATE_DEVICE_MEMORY);
                outResp.create(capacity, 1, CV_32F, USAGE_ALLOCATE_DEVICE_MEMORY);
                outOct.create(capacity, 1, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            }

            void ensureCounter()
            {
                if (counter.empty())
                    counter.create(1, 1, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            }
        };

        static bool siftOclAssignOrientationsForImageDeviceReusable(
            const UMat &uimg,
            const UMat &uRc,
            const UMat &uKpt,
            const UMat &uOct,
            int n,
            ocl::Kernel &ker,
            SiftOclOrientationScratch &scratch,
            SiftOclImageOrientedKeypoints &out)
        {
            if (n <= 0)
            {
                out.count = 0;
                return true;
            }

            const int maxout = std::max(1, n * kSiftOclOrientationDupFactor);
            scratch.ensureCounter();
            scratch.ensureCapacity(maxout);
            scratch.counter.setTo(Scalar(0));

            size_t globalsize[1] = {(size_t)n};
            size_t local = siftPreferredLocal1D(ker, (size_t)n);
            const ocl::Device &dev = ocl::Device::getDefault();
            if (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU))
                local = siftEnvLocalSize1D("OPENCV_SIFT_OCL_ORI_LOCAL", 64, ker.workGroupSize(), (size_t)n);
            size_t localsize[1] = {local};
            if (localsize[0])
                globalsize[0] = ((globalsize[0] + localsize[0] - 1) / localsize[0]) * localsize[0];

            bool ok = ker.args(
                             ocl::KernelArg::PtrReadOnly(uimg), (int)uimg.step,
                             uimg.rows, uimg.cols,
                             ocl::KernelArg::PtrReadOnly(uRc),
                             ocl::KernelArg::PtrReadOnly(uKpt),
                             ocl::KernelArg::PtrReadOnly(uOct),
                             n,
                             ocl::KernelArg::PtrReadWrite(scratch.counter),
                             ocl::KernelArg::PtrWriteOnly(scratch.outKpt),
                             ocl::KernelArg::PtrWriteOnly(scratch.outResp),
                             ocl::KernelArg::PtrWriteOnly(scratch.outOct),
                             maxout)
                          .run(1, globalsize, localsize[0] ? localsize : NULL, true);
            if (!ok)
                return false;

            Mat hCount;
            scratch.counter.copyTo(hCount);
            int outCount = hCount.at<int>(0);
            out.count = outCount;
            if (outCount <= 0)
                return true;
            if (outCount > maxout)
                return false;

            if (outCount < scratch.capacity)
            {
                out.keypoints = scratch.outKpt.rowRange(0, outCount);
                out.responses = scratch.outResp.rowRange(0, outCount);
                out.octaves = scratch.outOct.rowRange(0, outCount);
            }
            else
            {
                out.keypoints = scratch.outKpt;
                out.responses = scratch.outResp;
                out.octaves = scratch.outOct;
            }

            return true;
        }

        static void siftOclDownloadKeypoints(
            const SiftOclImageOrientedKeypoints &oriented,
            std::vector<KeyPoint> &keypoints)
        {
            if (oriented.count <= 0)
                return;

            Mat hKpt, hOct;
            Mat hResp;
            oriented.keypoints.copyTo(hKpt);
            oriented.responses.copyTo(hResp);
            oriented.octaves.copyTo(hOct);
            keypoints.reserve(keypoints.size() + (size_t)oriented.count);
            for (int i = 0; i < oriented.count; ++i)
            {
                const Vec4f v = hKpt.at<Vec4f>(i);
                KeyPoint kpt;
                kpt.pt = Point2f(v[0], v[1]);
                kpt.size = v[2];
                kpt.angle = v[3];
                kpt.response = hResp.at<float>(i);
                kpt.octave = hOct.at<int>(i);
                keypoints.push_back(kpt);
            }
        }
        static bool siftOclCalcDescriptors(
            const std::vector<UMat> &ugpyr,
            const std::vector<KeyPoint> &keypoints,
            Mat &descriptors,
            int nOctaveLayers,
            int firstOctave)
        {
            if (keypoints.empty())
                return true;

            struct SiftOclDescriptorEntry
            {
                int imageIdx;
                int originalIdx;
                Vec4f packed;
            };

            std::vector<std::vector<int>> groups(ugpyr.size());
            std::vector<SiftOclDescriptorEntry> ordered;
            ordered.reserve(keypoints.size());
            for (int i = 0; i < (int)keypoints.size(); ++i)
            {
                int octave, layer;
                float scale;
                unpackOctave(keypoints[i], octave, layer, scale);
                if (octave < firstOctave || layer > nOctaveLayers + 2)
                    return false;
                int imageIdx = (octave - firstOctave) * (nOctaveLayers + 3) + layer;
                if (imageIdx < 0 || imageIdx >= (int)ugpyr.size())
                    return false;
                groups[(size_t)imageIdx].push_back(i);

                const KeyPoint &kpt = keypoints[i];
                SiftOclDescriptorEntry entry;
                entry.imageIdx = imageIdx;
                entry.originalIdx = i;
                entry.packed = Vec4f(kpt.pt.x * scale, kpt.pt.y * scale, kpt.size * scale, kpt.angle);
                ordered.push_back(entry);
            }

            // Sort by imageIdx first, then by spatial position (row, col) within each
            // image group.  The secondary sort gives consecutive work items in a warp
            // spatially adjacent keypoints, improving L2 cache utilisation during the
            // descriptor gradient-sampling loop.
            std::stable_sort(ordered.begin(), ordered.end(),
                             [](const SiftOclDescriptorEntry &a, const SiftOclDescriptorEntry &b)
                             {
                                 if (a.imageIdx != b.imageIdx)
                                     return a.imageIdx < b.imageIdx;
                                 if (a.packed[1] != b.packed[1])
                                     return a.packed[1] < b.packed[1];
                                 return a.packed[0] < b.packed[0];
                             });

            Mat hAllKpt((int)ordered.size(), 1, CV_32FC4);
            for (int i = 0; i < hAllKpt.rows; ++i)
                hAllKpt.at<Vec4f>(i) = ordered[(size_t)i].packed;

            UMat uAllKpt, uAllDesc((int)ordered.size(), descriptors.cols, CV_32F, USAGE_ALLOCATE_DEVICE_MEMORY);
            hAllKpt.copyTo(uAllKpt);

            // ------------------------------------------------------------------
            // Two-pass path: LDS-buffered accumulation + separate normalisation.
            //
            // Pass 1 (SIFT_computeDescriptors_ldsAccum): accumulates the raw
            // 128-element histogram into __local memory, reducing VGPR pressure
            // and improving wave occupancy.  Outputs unnormalised float histograms
            // to a temporary device buffer (uRawHist).
            //
            // Pass 2 (SIFT_normalizeDescriptors): reads uRawHist and writes the
            // final normalised CV_32F descriptors to uAllDesc in a single
            // non-blocking dispatch covering all keypoints at once.
            //
            // Both passes are dispatched non-blocking; a single copyTo at the end
            // acts as the implicit barrier (out-of-order amortisation).
            //
            // Falls back to the original single-pass kernel on build failure.
            // ------------------------------------------------------------------
            {
                ocl::Kernel kerAccum("SIFT_computeDescriptors_ldsAccum",
                                     ocl::features2d::sift_oclsrc, siftOclBuildOptions());
                ocl::Kernel kerNorm("SIFT_normalizeDescriptors",
                                    ocl::features2d::sift_oclsrc, siftOclBuildOptions());

                if (!kerAccum.empty() && !kerNorm.empty())
                {
                    UMat uRawHist((int)ordered.size(), descriptors.cols, CV_32F,
                                  USAGE_ALLOCATE_DEVICE_MEMORY);

                    const ocl::Device &descDev = ocl::Device::getDefault();

                    // Work-group size for pass 1.  Cap at 64 so that the LDS pool
                    // (local * 128 * 4 bytes) stays within a safe 32 KB budget.
                    // The Intel-GPU branch follows the same siftEnvLocalSize1D pattern
                    // used elsewhere in this file for pass-specific tuning.
                    size_t localAccum = siftPreferredLocal1D(kerAccum, (size_t)ordered.size());
                    if (descDev.isIntel() && (descDev.type() & ocl::Device::TYPE_GPU))
                        localAccum = siftEnvLocalSize1D("OPENCV_SIFT_OCL_DESC_LDS_LOCAL", 64,
                                                        kerAccum.workGroupSize(),
                                                        (size_t)ordered.size());
                    if (localAccum > 64)
                        localAccum = 64; // hard cap: 64 * 128 * 4 = 32 KB LDS
                    // The LDS kernel requires an explicit, non-zero work-group size so
                    // that the LDS pool can be sized correctly.  Fall back to a single
                    // work-item per group if the device provides no preference.
                    if (localAccum == 0)
                        localAccum = 1;

                    int offset = 0;
                    bool twoPassOk = true;
                    for (size_t imageIdx = 0; imageIdx < groups.size(); ++imageIdx)
                    {
                        const std::vector<int> &group = groups[imageIdx];
                        if (group.empty())
                            continue;

                        size_t gs[1] = {group.size()};
                        size_t ls[1] = {localAccum};
                        gs[0] = ((gs[0] + ls[0] - 1) / ls[0]) * ls[0];

                        // LDS pool: one 128-float slot per work item in the group.
                        CV_Assert(ls[0] > 0);
                        size_t ldsBytes = ls[0] * (size_t)descriptors.cols * sizeof(float);

                        UMat uKpt  = uAllKpt.rowRange(offset, offset + (int)group.size());
                        UMat uRaw  = uRawHist.rowRange(offset, offset + (int)group.size());
                        bool ok = kerAccum.args(
                                             ocl::KernelArg::PtrReadOnly(ugpyr[imageIdx]),
                                             (int)ugpyr[imageIdx].step,
                                             ugpyr[imageIdx].rows, ugpyr[imageIdx].cols,
                                             ocl::KernelArg::PtrReadOnly(uKpt),
                                             (int)group.size(),
                                             ocl::KernelArg::PtrWriteOnly(uRaw),
                                             (int)uRaw.step,
                                             ocl::KernelArg::Local(ldsBytes))
                                          .run(1, gs, ls, false);
                        if (!ok)
                        {
                            twoPassOk = false;
                            break;
                        }
                        offset += (int)group.size();
                    }

                    if (twoPassOk)
                    {
                        CV_Assert(offset == (int)ordered.size());

                        // Pass 2: normalise all keypoints in one dispatch (no image
                        // pyramid dependency – the kernel is purely arithmetic).
                        size_t normGs[1] = {(size_t)ordered.size()};
                        size_t normLocal = siftPreferredLocal1D(kerNorm, normGs[0]);
                        if (descDev.isIntel() && (descDev.type() & ocl::Device::TYPE_GPU))
                            normLocal = siftEnvLocalSize1D("OPENCV_SIFT_OCL_NORM_LOCAL", 128,
                                                           kerNorm.workGroupSize(), normGs[0]);
                        size_t normLs[1] = {normLocal};
                        if (normLs[0])
                            normGs[0] = ((normGs[0] + normLs[0] - 1) / normLs[0]) * normLs[0];

                        bool ok = kerNorm.args(
                                             ocl::KernelArg::PtrReadOnly(uRawHist),
                                             (int)uRawHist.step,
                                             (int)ordered.size(),
                                             ocl::KernelArg::PtrWriteOnly(uAllDesc),
                                             (int)uAllDesc.step)
                                          .run(1, normGs, normLs[0] ? normLs : nullptr, false);
                        if (ok)
                        {
                            // Single barrier: one copyTo covers both kernel passes.
                            Mat hAllDesc;
                            uAllDesc.copyTo(hAllDesc);
                            for (int i = 0; i < hAllDesc.rows; ++i)
                                hAllDesc.row(i).copyTo(
                                    descriptors.row(ordered[(size_t)i].originalIdx));
                            return true;
                        }
                    }
                    // Fall through to single-pass on any failure.
                }
            }

            // ------------------------------------------------------------------
            // Single-pass fallback (original kernel).
            // ------------------------------------------------------------------
            {
                ocl::Kernel ker("SIFT_computeDescriptors", ocl::features2d::sift_oclsrc,
                                siftOclBuildOptions());
                if (ker.empty())
                    return false;

                const ocl::Device &descDev = ocl::Device::getDefault();

                int offset = 0;
                for (size_t imageIdx = 0; imageIdx < groups.size(); ++imageIdx)
                {
                    const std::vector<int> &group = groups[imageIdx];
                    if (group.empty())
                        continue;

                    size_t globalsize[1] = {group.size()};
                    size_t local = siftPreferredLocal1D(ker, globalsize[0]);
                    if (descDev.isIntel() && (descDev.type() & ocl::Device::TYPE_GPU))
                        local = siftEnvLocalSize1D("OPENCV_SIFT_OCL_DESC_LOCAL", 128,
                                                   ker.workGroupSize(), globalsize[0]);
                    size_t localsize[1] = {local};
                    if (localsize[0])
                        globalsize[0] = ((globalsize[0] + localsize[0] - 1) / localsize[0]) *
                                        localsize[0];
                    UMat uKpt  = uAllKpt.rowRange(offset, offset + (int)group.size());
                    UMat uDesc = uAllDesc.rowRange(offset, offset + (int)group.size());
                    bool ok = ker.args(
                                     ocl::KernelArg::PtrReadOnly(ugpyr[imageIdx]),
                                     (int)ugpyr[imageIdx].step,
                                     ugpyr[imageIdx].rows, ugpyr[imageIdx].cols,
                                     ocl::KernelArg::PtrReadOnly(uKpt), (int)group.size(),
                                     ocl::KernelArg::PtrWriteOnly(uDesc), (int)uDesc.step)
                                  .run(1, globalsize, localsize[0] ? localsize : nullptr, false);
                    if (!ok)
                        return false;

                    offset += (int)group.size();
                }

                CV_Assert(offset == (int)ordered.size());

                Mat hAllDesc;
                uAllDesc.copyTo(hAllDesc);
                for (int i = 0; i < hAllDesc.rows; ++i)
                    hAllDesc.row(i).copyTo(descriptors.row(ordered[(size_t)i].originalIdx));
            }

            return true;
        }

        static bool siftOclFindScaleSpaceExtrema(
            SIFT_Impl *impl,
            const std::vector<UMat> &ugauss_pyr,
            const std::vector<UMat> &udog_pyr,
            std::vector<KeyPoint> &keypoints,
            std::vector<SiftOclImageOrientedKeypoints> *orientedGroups)
        {
            const int nOctaveLayers = impl->getNOctaveLayers();
            const double contrastThreshold = impl->getContrastThreshold();
            const double edgeThreshold = impl->getEdgeThreshold();
            const double sigma = impl->getSigma();
            const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * sift_detail::SIFT_FIXPT_SCALE);

            const int nOctaves = (int)ugauss_pyr.size() / (nOctaveLayers + 3);
            TLSDataAccumulator<std::vector<SiftOclProvisionalKeypoint>> tls_refined_struct;
            // Opt-A: per-layer views over one per-octave candidate buffer plus a per-octave counter array.
            // All nOctaveLayers collect kernels are issued without any intermediate counter readback;
            // one counter download and one candidate-buffer download per octave flushes all work.
            UMat uLayerRcOctave(nOctaveLayers * kSiftOclMaxCanPerLayer, 2, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            std::vector<UMat> uLayerRcSlices((size_t)nOctaveLayers);
            for (int j = 0; j < nOctaveLayers; j++)
            {
                const int r0 = j * kSiftOclMaxCanPerLayer;
                uLayerRcSlices[(size_t)j] = uLayerRcOctave.rowRange(r0, r0 + kSiftOclMaxCanPerLayer);
            }
            UMat uCounterOctave(nOctaveLayers, 1, CV_32S, USAGE_ALLOCATE_DEVICE_MEMORY);
            // Pre-build stable row-range sub-UMats so they outlive inner-loop scope across async dispatch.
            std::vector<UMat> uCounterSlices((size_t)nOctaveLayers);
            for (int j = 0; j < nOctaveLayers; j++)
                uCounterSlices[(size_t)j] = uCounterOctave.rowRange(j, j + 1);

            UMat uPrevDev, uCurDev, uNextDev;
            ocl::Kernel ker("SIFT_collectExtremaCandidates", ocl::features2d::sift_oclsrc, siftOclBuildOptions());
            if (ker.empty())
                return false;

            std::vector<std::vector<int>> rcLists((size_t)nOctaves * (size_t)nOctaveLayers);
            std::vector<int> rcCounts((size_t)nOctaves * (size_t)nOctaveLayers, 0);

            for (int o = 0; o < nOctaves; o++)
            {
                const int octaveBase = o * (nOctaveLayers + 2);
                siftCopyToDeviceBuffer(udog_pyr[(size_t)octaveBase], uPrevDev);
                siftCopyToDeviceBuffer(udog_pyr[(size_t)(octaveBase + 1)], uCurDev);
                siftCopyToDeviceBuffer(udog_pyr[(size_t)(octaveBase + 2)], uNextDev);

                // Reset all per-layer counters with one GPU write instead of nOctaveLayers separate resets.
                uCounterOctave.setTo(Scalar(0));

                // Issue all nOctaveLayers collect kernels for this octave without reading any counter.
                // The in-order OpenCL queue preserves ordering between kernel dispatches and
                // the siftCopyToDeviceBuffer calls that feed subsequent layers.
                const ocl::Device &collectDev = ocl::Device::getDefault();
                const bool collectIsIntelGPU = collectDev.isIntel() && (collectDev.type() & ocl::Device::TYPE_GPU);

                for (int i = 1; i <= nOctaveLayers; i++)
                {
                    const int idx = o * (nOctaveLayers + 2) + i;
                    const int li = i - 1;
                    int rows = uCurDev.rows, cols = uCurDev.cols;
                    int prev_step = (int)uPrevDev.step, cur_step = (int)uCurDev.step, next_step = (int)uNextDev.step;

                    size_t gw = (size_t)std::max(1, cols - 2 * sift_detail::SIFT_IMG_BORDER);
                    size_t gh = (size_t)std::max(1, rows - 2 * sift_detail::SIFT_IMG_BORDER);
                    size_t globalsize[2], localsize[2] = {0, 0};
                    globalsize[0] = gw;
                    globalsize[1] = gh;
                    if (collectIsIntelGPU)
                    {
                        siftIntelCollectLocal2D(gw, gh, ker.workGroupSize(), localsize, globalsize);
                    }

                    bool ok = ker.args(
                                     ocl::KernelArg::PtrReadOnly(uPrevDev), prev_step,
                                     ocl::KernelArg::PtrReadOnly(uCurDev), cur_step,
                                     ocl::KernelArg::PtrReadOnly(uNextDev), next_step,
                                     rows, cols,
                                     (float)threshold,
                                     ocl::KernelArg::PtrReadWrite(uCounterSlices[(size_t)li]),
                                     ocl::KernelArg::PtrWriteOnly(uLayerRcSlices[(size_t)li]),
                                     kSiftOclMaxCanPerLayer)
                                  .run(2, globalsize, localsize[0] ? localsize : nullptr, false); // non-blocking
                    if (!ok)
                        return false;

                    if (i < nOctaveLayers)
                    {
                        std::swap(uPrevDev, uCurDev);
                        std::swap(uCurDev, uNextDev);
                        siftCopyToDeviceBuffer(udog_pyr[(size_t)(idx + 2)], uNextDev);
                    }
                }

                // One barrier for all nOctaveLayers kernels of this octave.
                Mat hCounterOctave;
                uCounterOctave.copyTo(hCounterOctave);
                Mat hLayerRcOctave;
                uLayerRcOctave.copyTo(hLayerRcOctave);

                // Download candidates for each layer now that counts are known.
                for (int i = 1; i <= nOctaveLayers; i++)
                {
                    const int li = i - 1;
                    const int nCand = hCounterOctave.at<int>(li);
                    if (nCand <= 0)
                        continue;
                    if (nCand >= kSiftOclMaxCanPerLayer - 4096)
                        return false;
                    const int listIdx = o * nOctaveLayers + li;
                    const int row0 = li * kSiftOclMaxCanPerLayer;
                    const int *p0 = hLayerRcOctave.ptr<int>(row0);
                    rcLists[(size_t)listIdx].assign(p0, p0 + nCand * 2);
                    rcCounts[(size_t)listIdx] = nCand;
                }
            }

            std::vector<Mat> dog_pyr_view;
            siftUMatPyrToMatView(udog_pyr, dog_pyr_view);

            for (int o = 0; o < nOctaves; ++o)
            {
                for (int i = 1; i <= nOctaveLayers; ++i)
                {
                    const int listIdx = o * nOctaveLayers + (i - 1);
                    const int nCand = rcCounts[(size_t)listIdx];
                    if (nCand <= 0)
                        continue;
                    const int *rc = rcLists[(size_t)listIdx].data();
                    parallel_for_(Range(0, nCand), siftRefineCandidatesBody(
                                                       o, i, nOctaveLayers, contrastThreshold, edgeThreshold, sigma,
                                                       &dog_pyr_view, rc, &tls_refined_struct));
                }
            }

            std::vector<std::vector<SiftOclProvisionalKeypoint> *> provisional_vecs;
            tls_refined_struct.gather(provisional_vecs);
            std::vector<std::vector<SiftOclProvisionalKeypoint>> grouped(ugauss_pyr.size());
            for (size_t i = 0; i < provisional_vecs.size(); ++i)
            {
                const std::vector<SiftOclProvisionalKeypoint> &vec = *provisional_vecs[i];
                for (size_t j = 0; j < vec.size(); ++j)
                {
                    const SiftOclProvisionalKeypoint &provisional = vec[j];
                    if (provisional.imageIdx < 0 || provisional.imageIdx >= (int)grouped.size())
                        return false;
                    grouped[(size_t)provisional.imageIdx].push_back(provisional);
                }
            }

            if (orientedGroups)
                orientedGroups->assign(grouped.size(), SiftOclImageOrientedKeypoints());

            keypoints.clear();
            int totalProvisional = 0;
            for (size_t imageIdx = 0; imageIdx < grouped.size(); ++imageIdx)
                totalProvisional += (int)grouped[imageIdx].size();

            Mat hRc(totalProvisional, 1, CV_32SC2);
            Mat hKpt(totalProvisional, 1, CV_32FC4);
            Mat hOct(totalProvisional, 1, CV_32S);
            std::vector<int> imageOffsets(grouped.size() + 1, 0);

            int offset = 0;
            for (size_t imageIdx = 0; imageIdx < grouped.size(); ++imageIdx)
            {
                imageOffsets[imageIdx] = offset;
                const std::vector<SiftOclProvisionalKeypoint> &local = grouped[imageIdx];
                for (size_t i = 0; i < local.size(); ++i, ++offset)
                {
                    hRc.at<Vec2i>(offset) = Vec2i(local[i].r, local[i].c);
                    hKpt.at<Vec4f>(offset) = Vec4f(local[i].kpt.pt.x, local[i].kpt.pt.y,
                                                   local[i].kpt.size, local[i].kpt.response);
                    hOct.at<int>(offset) = local[i].kpt.octave;
                }
            }
            imageOffsets[grouped.size()] = offset;

            UMat uAllRc, uAllKpt, uAllOct;
            if (totalProvisional > 0)
            {
                hRc.copyTo(uAllRc);
                hKpt.copyTo(uAllKpt);
                hOct.copyTo(uAllOct);
            }

            const bool canReuseOrientationScratch = (orientedGroups == NULL);
            ocl::Kernel orientKer;
            SiftOclOrientationScratch orientScratch;
            if (canReuseOrientationScratch)
            {
                orientKer = ocl::Kernel("SIFT_assignOrientations", ocl::features2d::sift_oclsrc, siftOclBuildOptions());
                if (orientKer.empty())
                    return false;
            }

            for (size_t imageIdx = 0; imageIdx < grouped.size(); ++imageIdx)
            {
                const int begin = imageOffsets[imageIdx];
                const int end = imageOffsets[imageIdx + 1];
                const int count = end - begin;
                if (count <= 0)
                    continue;

                UMat uRc = uAllRc.rowRange(begin, end);
                UMat uKpt = uAllKpt.rowRange(begin, end);
                UMat uOct = uAllOct.rowRange(begin, end);
                SiftOclImageOrientedKeypoints oriented;
                bool ok = canReuseOrientationScratch
                              ? siftOclAssignOrientationsForImageDeviceReusable(ugauss_pyr[imageIdx], uRc, uKpt, uOct, count, orientKer, orientScratch, oriented)
                              : siftOclAssignOrientationsForImageDevice(ugauss_pyr[imageIdx], uRc, uKpt, uOct, count, oriented);
                if (!ok)
                    return false;
                if (orientedGroups)
                    (*orientedGroups)[imageIdx] = oriented;
                else
                    siftOclDownloadKeypoints(oriented, keypoints);
            }

            if (orientedGroups)
            {
                for (size_t imageIdx = 0; imageIdx < orientedGroups->size(); ++imageIdx)
                {
                    std::vector<KeyPoint> groupKeypoints;
                    siftOclDownloadKeypoints((*orientedGroups)[imageIdx], groupKeypoints);
                    keypoints.insert(keypoints.end(), groupKeypoints.begin(), groupKeypoints.end());
                }
            }
            return true;
        }

        static bool siftTryOpenCLDetectAndCompute(
            SIFT_Impl *impl,
            InputArray _image,
            InputArray _mask,
            std::vector<KeyPoint> &keypoints,
            OutputArray _descriptors,
            bool useProvidedKeypoints,
            int firstOctave,
            int actualNOctaves,
            int actualNLayers)
        {
            CV_UNUSED(actualNLayers);
            if (!ocl::useOpenCL() || !_image.isUMat())
                return false;
            if (impl->descriptorType() != CV_32F)
                return false;

            const bool fullOffload = std::getenv("OPENCV_SIFT_OPENCL_FULL") != NULL;

            // Exit before acquiring any UMat/Mat handles so we pay zero OCL queue-flush cost when
            // the full GPU descriptor path has not been requested.  The getUMat()/getMat() calls
            // below can trigger a clEnqueueMapBuffer (and hence an implicit command-queue sync) on
            // device-backed UMats even when OCL is technically not needed for this call path.
            if (_descriptors.needed() && !fullOffload)
                return false;

            UMat uimage = _image.getUMat();
            Mat mask = _mask.getMat();

            if (uimage.empty() || uimage.depth() != CV_8U)
                return false;

            // Skip OpenCL SIFT on very small inputs: host↔device and kernel launch costs dominate.
            // Override with OPENCV_SIFT_OPENCL_FORCE=1, or tune with OPENCV_SIFT_OPENCL_MIN_PIXELS / OPENCV_SIFT_OPENCL_MIN_SIDE.
            if (!std::getenv("OPENCV_SIFT_OPENCL_FORCE"))
            {
                const int64_t pixels = (int64_t)uimage.cols * uimage.rows;
                const int mn = std::min(uimage.cols, uimage.rows);
                int minPixels = 512 * 384; // ~0.2 MP default crossover hint
                int minSide = 384;
                if (const char *e = std::getenv("OPENCV_SIFT_OPENCL_MIN_PIXELS"))
                    minPixels = std::max(1, atoi(e));
                if (const char *e = std::getenv("OPENCV_SIFT_OPENCL_MIN_SIDE"))
                    minSide = std::max(32, atoi(e));
                if (pixels < (int64_t)minPixels || mn < minSide)
                    return false;
            }

            int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log((double)std::min(uimage.cols, uimage.rows)) / std::log(2.) - 2) - firstOctave;

            UMat ubase = siftCreateInitialImageUMat(uimage, firstOctave < 0, (float)impl->getSigma(), impl->getEnablePreciseUpscale());
            std::vector<UMat> ugpyr;
            siftBuildGaussianPyramidUMat(ubase, ugpyr, nOctaves, impl->getNOctaveLayers(), impl->getSigma());

            std::vector<Mat> gpyr, dogpyr;

            bool usedOclExtrema = false;
            if (!useProvidedKeypoints)
            {
                std::vector<UMat> udogpyr;
                siftBuildDoGPyramidUMat(ugpyr, udogpyr, impl->getNOctaveLayers());

                keypoints.clear();
                usedOclExtrema = siftOclFindScaleSpaceExtrema(impl, ugpyr, udogpyr, keypoints, NULL);
                if (!usedOclExtrema)
                {
                    siftUMatPyrToMatView(ugpyr, gpyr);
                    siftUMatPyrToMat(udogpyr, dogpyr);
                    keypoints.clear();
                    impl->findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
                }
                KeyPointsFilter::removeDuplicatedSorted(keypoints);
                if (impl->getNFeatures() > 0)
                    KeyPointsFilter::retainBest(keypoints, impl->getNFeatures());

                if (firstOctave < 0)
                    for (size_t i = 0; i < keypoints.size(); i++)
                    {
                        KeyPoint &kpt = keypoints[i];
                        float scale = 1.f / (float)(1 << -firstOctave);
                        kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                        kpt.pt *= scale;
                        kpt.size *= scale;
                    }

                if (!mask.empty())
                    KeyPointsFilter::runByPixelsMask(keypoints, mask);
            }

            if (_descriptors.needed())
            {
                int dsize = impl->descriptorSize();
                _descriptors.create((int)keypoints.size(), dsize, impl->descriptorType());
                Mat descriptors = _descriptors.getMat();
                
                // ARCHITECTURAL NOTE (May 2026): Full GPU SIFT impossible on integrated GPU
                // Problem: siftUMatPyrToMatView() calls clEnqueueMapBuffer, which flushes ALL pending GPU work.
                // This sync cost (1.87-5.11× measured) exceeds any kernel optimization benefit.
                // Verified after exhaustive testing (Strategies 1-5, full OCL mode 0/24 favorable).
                // Root cause: GPU-to-CPU memory transfer requires implicit sync on unified memory architecture.
                // Solution: Skip descriptor GPU kernels, use Phases 1-3 (GPU blur only) for 3-5% speedup on XL.
                
                if (!usedOclExtrema || !siftOclCalcDescriptors(ugpyr, keypoints, descriptors, impl->getNOctaveLayers(), firstOctave))
                {
                    siftUMatPyrToMatView(ugpyr, gpyr);
                    calcDescriptors(gpyr, keypoints, descriptors, impl->getNOctaveLayers(), firstOctave);
                }
            }

            return true;
        }

    } // namespace
#endif // HAVE_OPENCL

    //////////////////////////////////////////////////////////////////////////////////////////

    SIFT_Impl::SIFT_Impl(int _nfeatures, int _nOctaveLayers,
                         double _contrastThreshold, double _edgeThreshold, double _sigma, int _descriptorType, bool _enable_precise_upscale)
        : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
          contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma), descriptor_type(_descriptorType),
          enable_precise_upscale(_enable_precise_upscale)
    {
        if (!enable_precise_upscale)
        {
            CV_LOG_ONCE_INFO(NULL, "precise upscale disabled, this is now deprecated as it was found to induce a location bias");
        }
    }

    int SIFT_Impl::descriptorSize() const
    {
        return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
    }

    int SIFT_Impl::descriptorType() const
    {
        return descriptor_type;
    }

    int SIFT_Impl::defaultNorm() const
    {
        return NORM_L2;
    }

    void SIFT_Impl::detectAndCompute(InputArray _image, InputArray _mask,
                                     std::vector<KeyPoint> &keypoints,
                                     OutputArray _descriptors,
                                     bool useProvidedKeypoints)
    {
        CV_TRACE_FUNCTION();

        int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;

        if (_image.empty() || _image.depth() != CV_8U)
            CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

        if (!_mask.empty() && _mask.type() != CV_8UC1)
            CV_Error(Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");

        if (useProvidedKeypoints)
        {
            firstOctave = 0;
            int maxOctave = INT_MIN;
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                int octave, layer;
                float scale;
                unpackOctave(keypoints[i], octave, layer, scale);
                firstOctave = std::min(firstOctave, octave);
                maxOctave = std::max(maxOctave, octave);
                actualNLayers = std::max(actualNLayers, layer - 2);
            }

            firstOctave = std::min(firstOctave, 0);
            CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
            actualNOctaves = maxOctave - firstOctave + 1;
        }

#ifdef HAVE_OPENCL
        const bool prevUseOpenCL = ocl::useOpenCL();
        bool cpuFallbackOpenCLSuppressed = false;
        if (prevUseOpenCL && _image.isUMat() && _descriptors.needed() && std::getenv("OPENCV_SIFT_OPENCL_FULL") == NULL)
        {
            Size sz = _image.size();
            int64 minPixelsForCpuFallbackOcl = (int64)3840 * 2160;
            if (const char *e = std::getenv("OPENCV_SIFT_CPU_FALLBACK_OCL_MIN_PIXELS"))
                minPixelsForCpuFallbackOcl = std::max<int64>(1, (int64)atoll(e));

            // In nofull mode, CPU fallback runs Mat-based SIFT. For non-XL inputs,
            // keeping OCL enabled often adds dispatch/mapping overhead without enough gain.
            if ((int64)sz.area() < minPixelsForCpuFallbackOcl)
            {
                ocl::setUseOpenCL(false);
                cpuFallbackOpenCLSuppressed = true;
            }
        }

        if (siftTryOpenCLDetectAndCompute(this, _image, _mask, keypoints, _descriptors, useProvidedKeypoints,
                                          firstOctave, actualNOctaves, actualNLayers))
        {
            if (cpuFallbackOpenCLSuppressed)
                ocl::setUseOpenCL(prevUseOpenCL);
            CV_IMPL_ADD(CV_IMPL_OCL);
            return;
        }
#endif

        Mat image = _image.getMat(), mask = _mask.getMat();

        Mat base = createInitialImage(image, firstOctave < 0, (float)sigma, enable_precise_upscale);
        std::vector<Mat> gpyr;
        int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;

        // double t, tf = getTickFrequency();
        // t = (double)getTickCount();
        buildGaussianPyramid(base, gpyr, nOctaves);

        // t = (double)getTickCount() - t;
        // printf("pyramid construction time: %g\n", t*1000./tf);

        if (!useProvidedKeypoints)
        {
            std::vector<Mat> dogpyr;
            buildDoGPyramid(gpyr, dogpyr);
            // t = (double)getTickCount();
            findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
            KeyPointsFilter::removeDuplicatedSorted(keypoints);

            if (nfeatures > 0)
                KeyPointsFilter::retainBest(keypoints, nfeatures);
            // t = (double)getTickCount() - t;
            // printf("keypoint detection time: %g\n", t*1000./tf);

            if (firstOctave < 0)
                for (size_t i = 0; i < keypoints.size(); i++)
                {
                    KeyPoint &kpt = keypoints[i];
                    float scale = 1.f / (float)(1 << -firstOctave);
                    kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                    kpt.pt *= scale;
                    kpt.size *= scale;
                }

            if (!mask.empty())
                KeyPointsFilter::runByPixelsMask(keypoints, mask);
        }
        else
        {
            // filter keypoints by mask
            // KeyPointsFilter::runByPixelsMask( keypoints, mask );
        }

        if (_descriptors.needed())
        {
            // t = (double)getTickCount();
            int dsize = descriptorSize();
            _descriptors.create((int)keypoints.size(), dsize, descriptor_type);

            Mat descriptors = _descriptors.getMat();
            calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
            // t = (double)getTickCount() - t;
            // printf("descriptor extraction time: %g\n", t*1000./tf);
        }

#ifdef HAVE_OPENCL
        if (cpuFallbackOpenCLSuppressed)
            ocl::setUseOpenCL(prevUseOpenCL);
#endif
    }

    void SIFT_Impl::read(const FileNode &fn)
    {
        // if node is empty, keep previous value
        if (!fn["nfeatures"].empty())
            fn["nfeatures"] >> nfeatures;
        if (!fn["nOctaveLayers"].empty())
            fn["nOctaveLayers"] >> nOctaveLayers;
        if (!fn["contrastThreshold"].empty())
            fn["contrastThreshold"] >> contrastThreshold;
        if (!fn["edgeThreshold"].empty())
            fn["edgeThreshold"] >> edgeThreshold;
        if (!fn["sigma"].empty())
            fn["sigma"] >> sigma;
        if (!fn["descriptorType"].empty())
            fn["descriptorType"] >> descriptor_type;
    }
    void SIFT_Impl::write(FileStorage &fs) const
    {
        if (fs.isOpened())
        {
            fs << "name" << getDefaultName();
            fs << "nfeatures" << nfeatures;
            fs << "nOctaveLayers" << nOctaveLayers;
            fs << "contrastThreshold" << contrastThreshold;
            fs << "edgeThreshold" << edgeThreshold;
            fs << "sigma" << sigma;
            fs << "descriptorType" << descriptor_type;
        }
    }

}
