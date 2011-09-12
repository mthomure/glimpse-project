/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "retinal_layer.h"
#include <cmath>
#include "util.h"

void ComputeMeanAndVariance(float* data, int size, float* mean, float* var) {
  *mean = 0;
  for (float* p = data; p < data + size; ++p) {
    *mean += *p;
  }
  *mean /= size;
  float sum2 = 0.0;
  float sumc = 0.0;
  for (float* p = data; p < data + size; ++p) {
    float delta = *p - *mean;
    sum2 += delta * delta;
    sumc += delta;
  }
  *var = (sum2 - sumc * sumc / size) / (size - 1);
}

void CProcessRetina(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output) {
  int max_oheight, max_owidth;
  CMaxOutputDimensions(kheight, kwidth,
      1.0,  // scaling
      input.w0, input.w1, &max_oheight, &max_owidth);
  int oheight = output.w0;
  int owidth = output.w1;
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);
  int pad = kwidth / 2;
  Array2D<float> cache(kheight, kwidth);
  for (int top = 0; top < oheight; ++top) {
    for (int left = 0; left < owidth; ++left) {
      // Load the cache
      for (int ky = 0; ky < kheight; ++ky) {
        for (int kx = 0; kx < kwidth; ++kx) {
          cache(ky, kx) = input(top + ky, left + kx);
        }
      }
      // Compute new center pixel
      float mean, var;
      ComputeMeanAndVariance(cache.Get(), cache.Size(), &mean, &var);
      float std = sqrt(var) + bias;
      output(top, left) = (cache(pad, pad) - mean) / std;
    }
  }
}

void CProcessRetinaSSE(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output) {
#ifdef __SSE__
  ASSERT_EQUALS(reinterpret_cast<long>(input.Get()) % SSE_VECTOR_WIDTH_BYTES,
      0);
  ASSERT_EQUALS(reinterpret_cast<long>(output.Get()) % SSE_VECTOR_WIDTH_BYTES,
      0);
  int max_oheight, max_owidth, multi_kwidth, lpad;
  CMaxOutputDimensionsSSE(kheight, kwidth,
      1.0,  // scaling
      input.w0, input.w1, &max_oheight, &max_owidth, &multi_kwidth, &lpad);
  int multi_kheight = kheight;
  ASSERT_TRUE(output.w0 <= max_oheight);
  ASSERT_TRUE(output.w1 <= max_owidth);

  // XXX this assumes that (multi_kheight = kwidth).
  int height_pad = multi_kheight / 2;
  // This is the horizontal offset from the left edge of the (padded) window
  // to the center of the first horizontal position. Note that this is not
  // just half the eventual kX value, since the horizontal padding is not
  // symmetric. This is guaranteed to be 4-byte aligned.
  int width_pad = kwidth / 2 + lpad;
  v4f inv_ksize_v = _mm_set1_ps(1.0 / float(kheight * kwidth));
  v4f sqr_bias_v = _mm_set1_ps(bias * bias);
  SSEArray3D mask(SSE_VECTOR_WIDTH, multi_kheight, multi_kwidth);
  Memset(&mask, 0);
  for (int step = 0; step < SSE_VECTOR_WIDTH; ++step) {
    for(int j = 0; j < multi_kheight; ++j) {
      for(int i = lpad + step; i < (lpad + step + kwidth); ++i) {
        mask(step, j, i) = 1;
      }
    }
  }
  int oheight = output.w0;
  int owidth = output.w1;
  int iwidth = input.w1;
  // Make sure each row is 16-byte aligned.
  ASSERT_EQUALS(iwidth % SSE_VECTOR_WIDTH, 0);

  // For each image y-location
  for ( int top = 0; top < oheight; ++top ) {
    // For each chunk of vector elements
    for ( int left = 0; left < owidth; left += SSE_VECTOR_WIDTH ) {

      //~ int image_j = top + image.pad;
      //~ int image_i = left + image.pad;

      // Sums for computing mean of image neighborhood.
      v4f sums[SSE_VECTOR_WIDTH];
      // Sums for computing second moment of image neighborhood. Each set of
      // vector elements contains partial sums for one kernel application. Note:
      // __m128-typed variables are 16-byte aligned by default.
      v4f squares[SSE_VECTOR_WIDTH];

      // For different horizontal shifts of kernel center (each value of "step"
      // applies same kernel at different location)
      for (int step = 0; step < SSE_VECTOR_WIDTH; ++step) {
        sums[step] = _mm_set_ps1(0.0f);
        squares[step] = _mm_set_ps1(0.0f);

        // Apply kernel at location (left + (step-2), top)
        for (int kj = 0; kj < multi_kheight; ++kj) {
          // Process four (contiguous) horizontal image locations in parallel
          for (int ki = 0; ki < multi_kwidth; ki += SSE_VECTOR_WIDTH) {
            // Lookup image pixel at current location
            int y = top + kj; // - multi_kheight / 2;
            int x = left + ki; // - window_center_xoffset;
            v4f* pixel_pv = reinterpret_cast<v4f*>(&input(y, x));
            v4f* mask_pv = reinterpret_cast<v4f*>(&mask(step, kj, ki));

            v4f pixel_v = *pixel_pv;
            v4f mask_v = *mask_pv;

            // Apply mask to image pixels
            v4f masked_pixel_v = pixel_v * mask_v;

            // Accumulate masked pixels
            sums[step] += masked_pixel_v;

            // Accumulate squares
            squares[step] += masked_pixel_v * masked_pixel_v;
          }
        } // end apply kernel
      } // end horizontal shifts

      // At this point, we have results for multiple image locations. However,
      // each of these have been computed in parallel, so results must be
      // combined.

      // Compute E[x] = (sum x_i) / N
      v4f mean_v = SwizzleAdd(sums) * inv_ksize_v;

      // Compute sigma^2 = sqrt( E[x^2] - E[x]^2 )
      // Add the bias before taking square root to avoid NaN result in
      // homogeneous regions.
      v4f sigma_v = _mm_sqrt_ps( SwizzleAdd(squares) * inv_ksize_v -
          mean_v * mean_v + sqr_bias_v );

      // Normalize the center pixel of the receptive field as
      //   x' = (x - mu) / sigma
      const v4f* center_pixel_pv = reinterpret_cast<const v4f*>(
          &input(top + height_pad, left + width_pad));
      v4f v = (*center_pixel_pv - mean_v) / sigma_v;
      _mm_store_ps( &output(top, left), v );
    }
  } // end for image locations
#else
  // SSE intrinsics unavailable.
  ASSERT_TRUE(false);
#endif  // __SSE__
}
