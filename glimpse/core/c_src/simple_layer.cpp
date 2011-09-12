/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "simple_layer.h"
#include <cmath>
#include "util.h"

#include <iostream>
using std::cout;

// Compute activation for a layer of S-units. Computes RBF as the Gaussian
// function applied to the normalized dot product of the input and kernel
// arrays. Thus, this function assumes each kernel has unit length.
// @param bias Used to alter the divisor when normalizing the input patch.
// @param beta Controls width of Gaussian envolope when matching S1 prototypes.
void CProcessSimpleLayer(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, float beta, int scaling,
    ArrayRef3D<float>& output) {
  int num_ibands = input.w0;
  int iheight = input.w1;
  int iwidth = input.w2;
  int num_obands = output.w0;
  int num_kernels = kernels.w0;
  int num_kbands = kernels.w1;
  ASSERT_EQUALS(num_ibands, num_kbands);
  ASSERT_EQUALS(num_kernels, num_obands);
  int kwidth = kernels.w2;
  int kheight = kernels.w3;
  int oheight = output.w1;
  int owidth = output.w2;
  int max_oheight, max_owidth;
  CMaxOutputDimensions(kheight, kwidth, scaling, iheight, iwidth, &max_oheight,
      &max_owidth);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);
  //const float scaling_constant = exp(-4.0 * beta);

  // for all filters
  for (int oband = 0; oband < num_kernels; ++oband) {
    // for all locations
    for (int top = 0; top < oheight; ++top) {
      for (int left = 0; left < owidth; ++left) {
        int rj = scaling * top;
        int ri = scaling * left;
        float squares = 0.0;
        float dotprod = 0.0;

        // for all kernel bands
        for (int iband = 0; iband < num_ibands; ++iband) {
          // Dot product of kernels and input neighborhood
          for (int kj = 0; kj < kheight; ++kj) {
            for (int ki = 0; ki < kwidth; ++ki) {
              float x = input(iband, rj + kj, ri + ki);
              float k = kernels(oband, iband, kj, ki);
              dotprod += k * x;
              squares += x * x;
            }
          }
        } // end for all kernel bands

        // Bias (usually by 1.0) to prevent noise amplification. Add the
        // bias before taking the square root to avoid ill-conditioned input.
        float input_norm = sqrt(squares + bias);
        // An alternative approach to avoid amplifying noise (used, e.g., by
        // Pinto et al) is to clip the divisor.
        // float input_norm;
        // if (squares < 1.0) {
          // input_norm = 1.0;
        // } else {
          // input_norm = sqrt(squares);
        // }

        // Input patch is not normalized. To compensate, divide dotprod by
        // norm of the input.
        dotprod /= input_norm;

        // We have ||k|| = ||x|| = 1. Thus, by Cauchy-Schwarz:
        //   -1 <= dotprod <= +1
        DEBUG_ASSERT(-1.0001 <= dotprod && dotprod <= 1.0001);

        // Radial basis function with unit vectors, given by:
        //   y = exp( -beta * ||X - P||^2 )
        // where X and P are the input and the prototype. Here
        // we use the identity:
        //   ||X - V||^2 = 2 ( 1 - <X,V> )
        // for unit vectors X and V, where <.,.> denotes the dot
        // product.
        float result = exp(-2.0 * beta * (1.0 - dotprod));

        // The above result lies in the closed interval
        // [exp(-4*beta), 1]. Thus, we rescale to get values in
        // [-1, 1].
        //result = 2.0 * (result - scaling_constant) /
            //(1.0 - scaling_constant) - 1.0;

        output(oband, top, left) = result;
      }
    } // end for all locations
  } // end for all filters
}


// At S1, PANN applies one kernel at four location in parallel. At S2, Steven
// instead applies four kernels to one location.

// Note: not a significant speedup over basic implementation. This may reflect
// a problem with the data alignment.

// It's assumed that input has been shifted by left_pad units to the right.
void CProcessSimpleLayerSSE_NoScaling(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, float beta,
    ArrayRef3D<float>& output) {
#ifdef __SSE__
  const int scaling = 1;
  // Make sure base address is 4-byte aligned
  ASSERT_EQUALS(reinterpret_cast<long>(input.Get()) % SSE_VECTOR_WIDTH_BYTES,
      0);
  ASSERT_EQUALS(reinterpret_cast<long>(output.Get()) % SSE_VECTOR_WIDTH_BYTES,
      0);
  int iheight = input.w1;
  int iwidth = input.w2;
  int oheight = output.w1;
  int owidth = output.w2;
  // Make sure each row is 4-byte aligned.
  ASSERT_EQUALS(iwidth % SSE_VECTOR_WIDTH, 0);
  ASSERT_EQUALS(owidth % SSE_VECTOR_WIDTH, 0);
  int kwidth = kernels.w2;
  int kheight = kernels.w3;
  int max_oheight, max_owidth, multi_kwidth, left_pad;
  CMaxOutputDimensionsSSE(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth, &multi_kwidth, &left_pad);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);
  int num_in_bands = input.w0;
  int num_out_bands = output.w0;
  int num_kernels = kernels.w0;
  int num_kernel_bands = kernels.w1;
  ASSERT_EQUALS(num_in_bands, num_kernel_bands);
  ASSERT_EQUALS(num_kernels, num_out_bands);

  int multi_kheight = kheight;
  v4f bias_v = _mm_set_ps1(bias);

  SSEArray4D mask(SSE_VECTOR_WIDTH, num_kernel_bands, multi_kheight,
      multi_kwidth);
  SSEArray4D multi_kernel(SSE_VECTOR_WIDTH, num_kernel_bands, multi_kheight,
      multi_kwidth);
  // Mask off locations that don't correspond to kernel values.
  Memset(&mask, 0);
  for (int step = 0; step < SSE_VECTOR_WIDTH; ++step) {
    for (int band = 0; band < num_kernel_bands; ++band) {
      for (int j = 0; j < multi_kheight; ++j) {
        for (int i = 0; i < kwidth; ++i) {
          mask(step, band, j, i + left_pad + step) = 1;
        }
      }
    }
  }

  // For each kernel/output band
  for (int kidx = 0; kidx < num_kernels; ++kidx) {
    // Create multi-kernel, which is just copies of given kernel at four
    // horizontal locations.
    Memset(&multi_kernel, 0);
    for (int step = 0; step < 4; ++step) {
      for (int kb = 0; kb < num_kernel_bands; ++kb) {
        for (int kj = 0; kj < kheight; ++kj) {
          for (int ki = 0; ki < kwidth; ++ki) {
            multi_kernel(step, kb, kj, ki + left_pad + step) =
                kernels(kidx, kb, kj, ki);
          }
        }
      }
    }

    // For each image y-location
    for (int top = 0; top < oheight; ++top) {
      int sj = scaling * top;

      // For each chunk of vector elements
      for (int left = 0; left < owidth; left += SSE_VECTOR_WIDTH) {
        v4f sums[SSE_VECTOR_WIDTH];
        v4f squares[SSE_VECTOR_WIDTH];
        v4f dotprods[SSE_VECTOR_WIDTH];

        // For different horizontal shifts of kernel center (each value of
        // "step" applies same kernel at different location)
        for (int step = 0; step < SSE_VECTOR_WIDTH; ++step) {
          sums[step] = _mm_set_ps1(0.0f);
          squares[step] = _mm_set_ps1(0.0f);
          dotprods[step] = _mm_set_ps1(0.0f);

          // Map X location from output to input coordinates.
          int si = scaling * left;

          // Offset the X location to start of vector (in input coordinates)
          // corresponding to this horizontal step. Note that, due to operand
          // types, this uses integer division.
          si += SSE_VECTOR_WIDTH * ((scaling * step) / SSE_VECTOR_WIDTH);

          // Element offset (in input coordinates) within vector for this
          // horizontal step.
          int input_step = (scaling * step) % SSE_VECTOR_WIDTH;

          // Apply element of multi-kernel
          for (int kb = 0; kb < num_kernel_bands; ++kb) {
            for (int kj = 0; kj < multi_kheight; ++kj) {
              // Process four (contiguous) horizontal image locations in
              // parallel.
              for (int ki = 0; ki < multi_kwidth; ki += SSE_VECTOR_WIDTH) {
                // Apply mask to retina activities
                v4f* activity_pv = reinterpret_cast<v4f*>(
                    &input(kb, sj + kj, si + ki));
                v4f* mask_pv = reinterpret_cast<v4f*>(
                    &mask(input_step, kb, kj, ki));
                v4f masked_activity_v = *mask_pv * *activity_pv;

                // Accumulate masked activities
                sums[step] += masked_activity_v;

                // Accumulate squares
                squares[step] += masked_activity_v * masked_activity_v;

                // Accumulate dot products
                v4f* multi_kernel_pv = reinterpret_cast<v4f*>(
                    &multi_kernel(input_step, kb, kj, ki));
                dotprods[step] += *multi_kernel_pv * masked_activity_v;
              }
            }
          } // End apply element of multi-kernel
        } // End for each step

        // At this point, we have results for four image locations. However,
        // each of these have been computed in parallel, so results must be
        // combined.

        v4f input_norm = SwizzleAdd(squares) + bias_v;
        input_norm = _mm_sqrt_ps(input_norm);

        // Normalized dot product
        v4f v = SwizzleAdd(dotprods) / input_norm;
        _mm_store_ps( &output(kidx, top, left), v );
      }
    } // End for each image location

    // Apply exponential function to results for current output band
    for (int j = 0; j < oheight; ++j) {
      for (int i = 0; i < owidth; ++i) {
        float* d = &output(kidx, j, i);
        *d = exp(-2.0 * beta * (1.0 - *d));
      }
    }
  } // End for each output band

#else
  // SSE intrinsics unavailable.
  ASSERT_TRUE(false);
#endif  // __SSE__
}

