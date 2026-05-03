// This file is part of OpenCV project. It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.

#ifndef OPENCV_FEATURES2D_SRC_SIFT_DETAIL_HPP
#define OPENCV_FEATURES2D_SRC_SIFT_DETAIL_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

namespace cv {
namespace sift_detail {

// Keep in sync with sift.simd.hpp (DoG_TYPE_SHORT == 0)
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
static const int SIFT_MAX_INTERP_STEPS = 5;
static const int SIFT_IMG_BORDER = 5;
static const int SIFT_ORI_HIST_BINS = 36;
static const float SIFT_ORI_SIG_FCTR = 1.5f;
static const float SIFT_ORI_RADIUS = 4.5f;
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// Computes a gradient orientation histogram at a specified pixel (shared by CPU SIMD path and OpenCL post-processing).
inline
float calcOrientationHist(
        const Mat& img, Point pt, int radius,
        float sigma, float* hist, int n
)
{
    CV_TRACE_FUNCTION();

    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);

    cv::utils::BufferArea area;
    float *X = 0, *Y = 0, *Mag, *Ori = 0, *W = 0, *temphist = 0;
    area.allocate(X, len, CV_SIMD_WIDTH);
    area.allocate(Y, len, CV_SIMD_WIDTH);
    area.allocate(Ori, len, CV_SIMD_WIDTH);
    area.allocate(W, len, CV_SIMD_WIDTH);
    area.allocate(temphist, n+4, CV_SIMD_WIDTH);
    area.commit();
    temphist += 2;
    Mag = X;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vecsize = VTraits<v_float32>::vlanes();
    v_float32 nd360 = vx_setall_f32(n/360.f);
    v_int32 __n = vx_setall_s32(n);
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) bin_buf[VTraits<v_float32>::max_nlanes];
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) w_mul_mag_buf[VTraits<v_float32>::max_nlanes];

    for( ; k <= len - vecsize; k += vecsize )
    {
        v_float32 w = vx_load_aligned( W + k );
        v_float32 mag = vx_load_aligned( Mag + k );
        v_float32 ori = vx_load_aligned( Ori + k );
        v_int32 bin = v_round( v_mul(nd360, ori) );

        bin = v_select(v_ge(bin, __n), v_sub(bin, __n), bin);
        bin = v_select(v_lt(bin, vx_setzero_s32()), v_add(bin, __n), bin);

        w = v_mul(w, mag);
        v_store_aligned(bin_buf, bin);
        v_store_aligned(w_mul_mag_buf, w);
        for(int vi = 0; vi < vecsize; vi++)
        {
            temphist[bin_buf[vi]] += w_mul_mag_buf[vi];
        }
    }
#endif
    for( ; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 d_1_16 = vx_setall_f32(1.f/16.f);
    v_float32 d_4_16 = vx_setall_f32(4.f/16.f);
    v_float32 d_6_16 = vx_setall_f32(6.f/16.f);
    for( ; i <= n - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes() )
    {
        v_float32 tn2 = vx_load_aligned(temphist + i-2);
        v_float32 tn1 = vx_load(temphist + i-1);
        v_float32 t0 = vx_load(temphist + i);
        v_float32 t1 = vx_load(temphist + i+1);
        v_float32 t2 = vx_load(temphist + i+2);
        v_float32 _hist = v_fma(v_add(tn2, t2), d_1_16,
            v_fma(v_add(tn1, t1), d_4_16, v_mul(t0, d_6_16)));
        v_store(hist + i, _hist);
    }
#endif
    for( ; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}


inline
bool adjustLocalExtrema(
        const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
        int& layer, int& r, int& c, int nOctaveLayers,
        float contrastThreshold, float edgeThreshold, float sigma
)
{
    CV_TRACE_FUNCTION();

    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}

} // namespace sift_detail
} // namespace cv

#endif // OPENCV_FEATURES2D_SRC_SIFT_DETAIL_HPP
