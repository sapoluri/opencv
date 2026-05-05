// This file is part of OpenCV project. It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.

#ifdef cl_khr_global_int32_base_atomics
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#define SIFT_IMG_BORDER 5

inline float read_dog(__global const uchar* base, int step_bytes, int r, int c)
{
    return *(__global const float*)(base + (size_t)r * (size_t)step_bytes + (size_t)c * sizeof(float));
}

inline float read_dog_layer(__global const uchar* base, int step_bytes, int layer_rows, int layer, int r, int c)
{
    int rr = layer * layer_rows + r;
    return *(__global const float*)(base + (size_t)rr * (size_t)step_bytes + (size_t)c * sizeof(float));
}

// First-pass DoG extrema test (matches scalar path in sift.simd.hpp).
__kernel void SIFT_collectExtremaCandidates(
    __global const uchar* restrict prev_base, int prev_step,
    __global const uchar* restrict cur_base, int cur_step,
    __global const uchar* restrict next_base, int next_step,
    int rows, int cols,
    float threshold,
    volatile __global int* counter,
    __global int* restrict out_rc,
    int maxout
)
{
    int c = (int)get_global_id(0) + SIFT_IMG_BORDER;
    int r = (int)get_global_id(1) + SIFT_IMG_BORDER;
    if (r >= rows - SIFT_IMG_BORDER || c >= cols - SIFT_IMG_BORDER)
        return;

    float val = read_dog(cur_base, cur_step, r, c);
    if (fabs(val) <= threshold)
        return;

    float _00 = read_dog(cur_base, cur_step, r - 1, c - 1);
    float _01 = read_dog(cur_base, cur_step, r - 1, c);
    float _02 = read_dog(cur_base, cur_step, r - 1, c + 1);
    float _10 = read_dog(cur_base, cur_step, r, c - 1);
    float _12 = read_dog(cur_base, cur_step, r, c + 1);
    float _20 = read_dog(cur_base, cur_step, r + 1, c - 1);
    float _21 = read_dog(cur_base, cur_step, r + 1, c);
    float _22 = read_dog(cur_base, cur_step, r + 1, c + 1);

    bool calculate = false;
    if (val > 0.f)
    {
        float vmax = fmax(fmax(fmax(_00, _01), fmax(_02, _10)), fmax(fmax(_12, _20), fmax(_21, _22)));
        if (val >= vmax)
        {
            _00 = read_dog(prev_base, prev_step, r - 1, c - 1);
            _01 = read_dog(prev_base, prev_step, r - 1, c);
            _02 = read_dog(prev_base, prev_step, r - 1, c + 1);
            _10 = read_dog(prev_base, prev_step, r, c - 1);
            _12 = read_dog(prev_base, prev_step, r, c + 1);
            _20 = read_dog(prev_base, prev_step, r + 1, c - 1);
            _21 = read_dog(prev_base, prev_step, r + 1, c);
            _22 = read_dog(prev_base, prev_step, r + 1, c + 1);
            vmax = fmax(fmax(fmax(_00, _01), fmax(_02, _10)), fmax(fmax(_12, _20), fmax(_21, _22)));
            if (val >= vmax)
            {
                _00 = read_dog(next_base, next_step, r - 1, c - 1);
                _01 = read_dog(next_base, next_step, r - 1, c);
                _02 = read_dog(next_base, next_step, r - 1, c + 1);
                _10 = read_dog(next_base, next_step, r, c - 1);
                _12 = read_dog(next_base, next_step, r, c + 1);
                _20 = read_dog(next_base, next_step, r + 1, c - 1);
                _21 = read_dog(next_base, next_step, r + 1, c);
                _22 = read_dog(next_base, next_step, r + 1, c + 1);
                vmax = fmax(fmax(fmax(_00, _01), fmax(_02, _10)), fmax(fmax(_12, _20), fmax(_21, _22)));
                if (val >= vmax)
                {
                    float _11p = read_dog(prev_base, prev_step, r, c);
                    float _11n = read_dog(next_base, next_step, r, c);
                    calculate = (val >= fmax(_11p, _11n));
                }
            }
        }
    }
    else
    {
        float vmin = fmin(fmin(fmin(_00, _01), fmin(_02, _10)), fmin(fmin(_12, _20), fmin(_21, _22)));
        if (val <= vmin)
        {
            _00 = read_dog(prev_base, prev_step, r - 1, c - 1);
            _01 = read_dog(prev_base, prev_step, r - 1, c);
            _02 = read_dog(prev_base, prev_step, r - 1, c + 1);
            _10 = read_dog(prev_base, prev_step, r, c - 1);
            _12 = read_dog(prev_base, prev_step, r, c + 1);
            _20 = read_dog(prev_base, prev_step, r + 1, c - 1);
            _21 = read_dog(prev_base, prev_step, r + 1, c);
            _22 = read_dog(prev_base, prev_step, r + 1, c + 1);
            vmin = fmin(fmin(fmin(_00, _01), fmin(_02, _10)), fmin(fmin(_12, _20), fmin(_21, _22)));
            if (val <= vmin)
            {
                _00 = read_dog(next_base, next_step, r - 1, c - 1);
                _01 = read_dog(next_base, next_step, r - 1, c);
                _02 = read_dog(next_base, next_step, r - 1, c + 1);
                _10 = read_dog(next_base, next_step, r, c - 1);
                _12 = read_dog(next_base, next_step, r, c + 1);
                _20 = read_dog(next_base, next_step, r + 1, c - 1);
                _21 = read_dog(next_base, next_step, r + 1, c);
                _22 = read_dog(next_base, next_step, r + 1, c + 1);
                vmin = fmin(fmin(fmin(_00, _01), fmin(_02, _10)), fmin(fmin(_12, _20), fmin(_21, _22)));
                if (val <= vmin)
                {
                    float _11p = read_dog(prev_base, prev_step, r, c);
                    float _11n = read_dog(next_base, next_step, r, c);
                    calculate = (val <= fmin(_11p, _11n));
                }
            }
        }
    }

    if (!calculate)
        return;

    int idx = atomic_inc(counter);
    if (idx < maxout)
    {
        out_rc[idx * 2] = r;
        out_rc[idx * 2 + 1] = c;
    }
}

// Subpixel refinement and edge/contrast rejection (matches sift_detail::adjustLocalExtrema logic).
__kernel void SIFT_refineExtremaCandidates(
    __global const uchar* restrict dog_base,
    int dog_step,
    int layer_rows,
    int rows,
    int cols,
    int nOctaveLayers,
    __global const int* restrict in_rc,
    int nCand,
    float contrastThreshold,
    float edgeThreshold,
    float sigma,
    int octv,
    int init_layer,
    __global int* restrict out_valid,
    __global int* restrict out_layer,
    __global int* restrict out_octave,
    __global float4* restrict out_kpt
)
{
    int k = (int)get_global_id(0);
    if (k >= nCand)
        return;

    const float img_scale = 1.0f / 255.0f;
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;

    int r = in_rc[k * 2 + 0];
    int c = in_rc[k * 2 + 1];
    int layer = init_layer;
    float xi = 0.0f, xr = 0.0f, xc = 0.0f, contr = 0.0f;
    int ok = 1;
    int it;

    for (it = 0; it < 5; ++it)
    {
        float v = read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c);
        float dDx = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c + 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c - 1)) * deriv_scale;
        float dDy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c)) * deriv_scale;
        float dDs = (read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r, c)) * deriv_scale;

        float v2 = v * 2.0f;
        float dxx = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c + 1) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c - 1) - v2) * second_deriv_scale;
        float dyy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c) - v2) * second_deriv_scale;
        float dss = (read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r, c) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r, c) - v2) * second_deriv_scale;
        float dxy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c + 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c - 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c + 1) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c - 1)) * cross_deriv_scale;
        float dxs = (read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r, c + 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r, c - 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r, c + 1) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r, c - 1)) * cross_deriv_scale;
        float dys = (read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r + 1, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r - 1, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r + 1, c) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r - 1, c)) * cross_deriv_scale;

        // Solve H*X=dD for X using adjugate/determinant.
        float det = dxx * (dyy * dss - dys * dys)
                  - dxy * (dxy * dss - dxs * dys)
                  + dxs * (dxy * dys - dxs * dyy);
        if (fabs(det) < 1e-20f)
        {
            ok = 0;
            break;
        }

        float invDet = 1.0f / det;
        float i00 =  (dyy * dss - dys * dys) * invDet;
        float i01 = -(dxy * dss - dxs * dys) * invDet;
        float i02 =  (dxy * dys - dxs * dyy) * invDet;
        float i11 =  (dxx * dss - dxs * dxs) * invDet;
        float i12 = -(dxx * dys - dxy * dxs) * invDet;
        float i22 =  (dxx * dyy - dxy * dxy) * invDet;

        float X0 = i00 * dDx + i01 * dDy + i02 * dDs;
        float X1 = i01 * dDx + i11 * dDy + i12 * dDs;
        float X2 = i02 * dDx + i12 * dDy + i22 * dDs;

        xc = -X0;
        xr = -X1;
        xi = -X2;

        if (fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
            break;

        if (fabs(xi) > 715827904.0f || fabs(xr) > 715827904.0f || fabs(xc) > 715827904.0f)
        {
            ok = 0;
            break;
        }

        c += convert_int_rte(xc);
        r += convert_int_rte(xr);
        layer += convert_int_rte(xi);

        if (layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= cols - SIFT_IMG_BORDER ||
            r < SIFT_IMG_BORDER || r >= rows - SIFT_IMG_BORDER)
        {
            ok = 0;
            break;
        }
    }

    if (!ok || it >= 5)
    {
        out_valid[k] = 0;
        return;
    }

    {
        float v = read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c);
        float dDx = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c + 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c - 1)) * deriv_scale;
        float dDy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c)) * deriv_scale;
        float dDs = (read_dog_layer(dog_base, dog_step, layer_rows, layer + 1, r, c) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer - 1, r, c)) * deriv_scale;
        float t = dDx * xc + dDy * xr + dDs * xi;

        contr = v * img_scale + t * 0.5f;
        if (fabs(contr) * nOctaveLayers < contrastThreshold)
        {
            out_valid[k] = 0;
            return;
        }

        float v2 = v * 2.0f;
        float dxx = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c + 1) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r, c - 1) - v2) * second_deriv_scale;
        float dyy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c) - v2) * second_deriv_scale;
        float dxy = (read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c + 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r + 1, c - 1) -
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c + 1) +
                     read_dog_layer(dog_base, dog_step, layer_rows, layer, r - 1, c - 1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det2 = dxx * dyy - dxy * dxy;
        if (det2 <= 0.0f || tr * tr * edgeThreshold >= (edgeThreshold + 1.0f) * (edgeThreshold + 1.0f) * det2)
        {
            out_valid[k] = 0;
            return;
        }
    }

    out_valid[k] = 1;
    out_layer[k] = layer;
    out_octave[k] = octv + (layer << 8) + (convert_int_rte((xi + 0.5f) * 255.0f) << 16);
    out_kpt[k] = (float4)(
        (c + xc) * (float)(1 << octv),
        (r + xr) * (float)(1 << octv),
        sigma * pow(2.0f, ((float)layer + xi) / (float)nOctaveLayers) * (float)(1 << octv) * 2.0f,
        fabs(contr));
}

#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5f
#define SIFT_ORI_RADIUS 4.5f
#define SIFT_ORI_PEAK_RATIO 0.8f
#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8
#define SIFT_DESCR_SCL_FCTR 3.0f
#define SIFT_DESCR_MAG_THR 0.2f
#define SIFT_INT_DESCR_FCTR 512.0f

inline float sift_angle_deg(float y, float x)
{
    float a = degrees(atan2(y, x));
    return a < 0.0f ? a + 360.0f : a;
}

__kernel void SIFT_assignOrientations(
    __global const uchar* restrict img_base,
    int img_step,
    int rows,
    int cols,
    __global const int2* restrict in_rc,
    __global const float4* restrict in_kpt,
    __global const int* restrict in_octave,
    int nCand,
    volatile __global int* counter,
    __global float4* restrict out_kpt,
    __global float* restrict out_response,
    __global int* restrict out_octave,
    int maxout)
{
    int idx = (int)get_global_id(0);
    if (idx >= nCand)
        return;

    int2 rc = in_rc[idx];
    int r0 = rc.x;
    int c0 = rc.y;
    float4 kp = in_kpt[idx];
    float scl_octv = kp.z * 0.5f;
    int radius = convert_int_rte(SIFT_ORI_RADIUS * scl_octv);
    float sigma = SIFT_ORI_SIG_FCTR * scl_octv;
    float expf_scale = -1.0f / (2.0f * sigma * sigma);
    float hist[SIFT_ORI_HIST_BINS];
    float smoothed[SIFT_ORI_HIST_BINS];

    for (int i = 0; i < SIFT_ORI_HIST_BINS; ++i)
        hist[i] = 0.0f;

    for (int i = -radius; i <= radius; ++i)
    {
        int y = r0 + i;
        if (y <= 0 || y >= rows - 1)
            continue;
        for (int j = -radius; j <= radius; ++j)
        {
            int x = c0 + j;
            if (x <= 0 || x >= cols - 1)
                continue;

            float dx = read_dog(img_base, img_step, y, x + 1) - read_dog(img_base, img_step, y, x - 1);
            float dy = read_dog(img_base, img_step, y - 1, x) - read_dog(img_base, img_step, y + 1, x);
            float w = exp((float)(i * i + j * j) * expf_scale);
            float mag = hypot(dx, dy);
            float ori = sift_angle_deg(dy, dx);
            int bin = convert_int_rte((SIFT_ORI_HIST_BINS / 360.0f) * ori);
            if (bin >= SIFT_ORI_HIST_BINS)
                bin -= SIFT_ORI_HIST_BINS;
            if (bin < 0)
                bin += SIFT_ORI_HIST_BINS;
            hist[bin] += w * mag;
        }
    }

    float maxval = 0.0f;
    for (int i = 0; i < SIFT_ORI_HIST_BINS; ++i)
    {
        float hm2 = hist[(i + SIFT_ORI_HIST_BINS - 2) % SIFT_ORI_HIST_BINS];
        float hm1 = hist[(i + SIFT_ORI_HIST_BINS - 1) % SIFT_ORI_HIST_BINS];
        float hp1 = hist[(i + 1) % SIFT_ORI_HIST_BINS];
        float hp2 = hist[(i + 2) % SIFT_ORI_HIST_BINS];
        float v = (hm2 + hp2) * (1.0f / 16.0f) + (hm1 + hp1) * (4.0f / 16.0f) + hist[i] * (6.0f / 16.0f);
        smoothed[i] = v;
        maxval = fmax(maxval, v);
    }

    float mag_thr = maxval * SIFT_ORI_PEAK_RATIO;
    for (int j = 0; j < SIFT_ORI_HIST_BINS; ++j)
    {
        int l = j > 0 ? j - 1 : SIFT_ORI_HIST_BINS - 1;
        int r = j < SIFT_ORI_HIST_BINS - 1 ? j + 1 : 0;
        float hj = smoothed[j];
        if (hj > smoothed[l] && hj > smoothed[r] && hj >= mag_thr)
        {
            float bin = (float)j + 0.5f * (smoothed[l] - smoothed[r]) / (smoothed[l] - 2.0f * hj + smoothed[r]);
            if (bin < 0.0f)
                bin += (float)SIFT_ORI_HIST_BINS;
            else if (bin >= (float)SIFT_ORI_HIST_BINS)
                bin -= (float)SIFT_ORI_HIST_BINS;

            float angle = 360.0f - (360.0f / (float)SIFT_ORI_HIST_BINS) * bin;
            if (fabs(angle - 360.0f) < 1e-6f)
                angle = 0.0f;

            int outidx = atomic_inc(counter);
            if (outidx < maxout)
            {
                out_kpt[outidx] = (float4)(kp.x, kp.y, kp.z, angle);
                out_response[outidx] = kp.w;
                out_octave[outidx] = in_octave[idx];
            }
        }
    }
}
__kernel void SIFT_computeDescriptors(
    __global const uchar* restrict img_base,
    int img_step,
    int rows,
    int cols,
    __global const float4* restrict in_kpt,
    int nKp,
    __global uchar* restrict out_base,
    int out_step)
{
    int idx = (int)get_global_id(0);
    if (idx >= nKp)
        return;

    float4 kp = in_kpt[idx];
    float pt_x = kp.x;
    float pt_y = kp.y;
    float scl = kp.z * 0.5f;
    float ori = 360.0f - kp.w;
    if (fabs(ori - 360.0f) < 1e-6f)
        ori = 0.0f;

    int pt_ix = convert_int_rte(pt_x);
    int pt_iy = convert_int_rte(pt_y);
    float cos_t = cos(radians(ori));
    float sin_t = sin(radians(ori));
    float bins_per_rad = SIFT_DESCR_HIST_BINS / 360.0f;
    float exp_scale = -1.0f / (SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = convert_int_rte(hist_width * 1.4142135623730951f * (SIFT_DESCR_WIDTH + 1) * 0.5f);
    int max_radius = convert_int_rte(sqrt((float)(cols * cols + rows * rows)));
    if (radius > max_radius)
        radius = max_radius;
    cos_t /= hist_width;
    sin_t /= hist_width;

    float rawDst[SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS];
    for (int i = 0; i < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++i)
        rawDst[i] = 0.0f;

    for (int i = -radius; i <= radius; ++i)
    {
        for (int j = -radius; j <= radius; ++j)
        {
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + SIFT_DESCR_WIDTH * 0.5f - 0.5f;
            float cbin = c_rot + SIFT_DESCR_WIDTH * 0.5f - 0.5f;
            int r = pt_iy + i;
            int c = pt_ix + j;
            if (!(rbin > -1.0f && rbin < SIFT_DESCR_WIDTH && cbin > -1.0f && cbin < SIFT_DESCR_WIDTH &&
                  r > 0 && r < rows - 1 && c > 0 && c < cols - 1))
                continue;

            float dx = read_dog(img_base, img_step, r, c + 1) - read_dog(img_base, img_step, r, c - 1);
            float dy = read_dog(img_base, img_step, r - 1, c) - read_dog(img_base, img_step, r + 1, c);
            float mag = hypot(dx, dy);
            float sample_ori = sift_angle_deg(dy, dx);
            float obin = (sample_ori - ori) * bins_per_rad;
            float w = exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);

            int r0 = convert_int_sat_rtn(floor(rbin));
            int c0 = convert_int_sat_rtn(floor(cbin));
            int o0 = convert_int_sat_rtn(floor(obin));
            rbin -= (float)r0;
            cbin -= (float)c0;
            obin -= (float)o0;

            while (o0 < 0)
                o0 += SIFT_DESCR_HIST_BINS;
            while (o0 >= SIFT_DESCR_HIST_BINS)
                o0 -= SIFT_DESCR_HIST_BINS;

            float v_r1 = mag * w * rbin;
            float v_r0 = mag * w - v_r1;
            float v_rc11 = v_r1 * cbin;
            float v_rc10 = v_r1 - v_rc11;
            float v_rc01 = v_r0 * cbin;
            float v_rc00 = v_r0 - v_rc01;
            float v_rco111 = v_rc11 * obin;
            float v_rco110 = v_rc11 - v_rco111;
            float v_rco101 = v_rc10 * obin;
            float v_rco100 = v_rc10 - v_rco101;
            float v_rco011 = v_rc01 * obin;
            float v_rco010 = v_rc01 - v_rco011;
            float v_rco001 = v_rc00 * obin;
            float v_rco000 = v_rc00 - v_rco001;

            int o1 = o0 + 1;
            if (o1 >= SIFT_DESCR_HIST_BINS)
                o1 -= SIFT_DESCR_HIST_BINS;

            if (r0 >= 0 && r0 < SIFT_DESCR_WIDTH)
            {
                if (c0 >= 0 && c0 < SIFT_DESCR_WIDTH)
                {
                    int base = (r0 * SIFT_DESCR_WIDTH + c0) * SIFT_DESCR_HIST_BINS;
                    rawDst[base + o0] += v_rco000;
                    rawDst[base + o1] += v_rco001;
                }
                if (c0 + 1 >= 0 && c0 + 1 < SIFT_DESCR_WIDTH)
                {
                    int base = (r0 * SIFT_DESCR_WIDTH + (c0 + 1)) * SIFT_DESCR_HIST_BINS;
                    rawDst[base + o0] += v_rco010;
                    rawDst[base + o1] += v_rco011;
                }
            }
            if (r0 + 1 >= 0 && r0 + 1 < SIFT_DESCR_WIDTH)
            {
                if (c0 >= 0 && c0 < SIFT_DESCR_WIDTH)
                {
                    int base = ((r0 + 1) * SIFT_DESCR_WIDTH + c0) * SIFT_DESCR_HIST_BINS;
                    rawDst[base + o0] += v_rco100;
                    rawDst[base + o1] += v_rco101;
                }
                if (c0 + 1 >= 0 && c0 + 1 < SIFT_DESCR_WIDTH)
                {
                    int base = ((r0 + 1) * SIFT_DESCR_WIDTH + (c0 + 1)) * SIFT_DESCR_HIST_BINS;
                    rawDst[base + o0] += v_rco110;
                    rawDst[base + o1] += v_rco111;
                }
            }
        }
    }

    float nrm2 = 0.0f;
    for (int i = 0; i < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++i)
        nrm2 += rawDst[i] * rawDst[i];
    float thr = sqrt(nrm2) * SIFT_DESCR_MAG_THR;
    nrm2 = 0.0f;
    for (int i = 0; i < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++i)
    {
        rawDst[i] = fmin(rawDst[i], thr);
        nrm2 += rawDst[i] * rawDst[i];
    }
    float scale = SIFT_INT_DESCR_FCTR / fmax(sqrt(nrm2), FLT_EPSILON);

    __global float* out_row = (__global float*)(out_base + (size_t)idx * (size_t)out_step);
    for (int i = 0; i < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++i)
        out_row[i] = clamp(rint(rawDst[i] * scale), 0.0f, 255.0f);
}
