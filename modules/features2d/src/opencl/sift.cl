// This file is part of OpenCV project. It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.

#ifdef cl_khr_global_int32_base_atomics
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#define SIFT_IMG_BORDER 5

inline float read_dog(__global const uchar* base, int step_bytes, int r, int c)
{
    return *(__global const float*)(base + (size_t)r * (size_t)step_bytes + (size_t)c * sizeof(float));
}

// First-pass DoG extrema test (matches scalar path in sift.simd.hpp).
__kernel void SIFT_collectExtremaCandidates(
    __global const uchar* prev_base, int prev_step,
    __global const uchar* cur_base, int cur_step,
    __global const uchar* next_base, int next_step,
    int rows, int cols,
    float threshold,
    volatile __global int* counter,
    __global int* out_rc,
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
