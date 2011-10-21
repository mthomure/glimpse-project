/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "filter.h"
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

void CContrastEnhance(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output) {
  int max_oheight, max_owidth;
  COutputMapShapeForInput(kheight, kwidth,
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
      float std = sqrt(var + bias);
      output(top, left) = (cache(pad, pad) - mean) / std;
    }
  }
}

void CDotProduct(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, int scaling, ArrayRef3D<float>& output) {
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
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);

  // for all filters
  for (int oband = 0; oband < num_kernels; ++oband) {
    // for all locations
    for (int top = 0; top < oheight; ++top) {
      for (int left = 0; left < owidth; ++left) {
        int rj = scaling * top;
        int ri = scaling * left;
        float dotprod = 0.0;
        // for all kernel bands
        for (int iband = 0; iband < num_ibands; ++iband) {
          // Dot product of kernels and input neighborhood
          for (int kj = 0; kj < kheight; ++kj) {
            for (int ki = 0; ki < kwidth; ++ki) {
              float x = input(iband, rj + kj, ri + ki);
              float k = kernels(oband, iband, kj, ki);
              dotprod += k * x;
            }
          }
        } // end for all kernel bands
        output(oband, top, left) = dotprod;
      }
    } // end for all locations
  } // end for all filters
}

void CNormDotProduct(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, int scaling,
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
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);

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
        //   float input_norm = sqrt(squares + bias);
        // An alternative approach to avoid amplifying noise (used, e.g., by
        // Pinto et al) is to clip the divisor.
        squares = (squares < bias ? bias : squares);
        float input_norm = sqrt(squares);

        // Input patch is not normalized. To compensate, divide dotprod by
        // norm of the input.
        dotprod /= input_norm;

        // We have ||k|| = ||x|| = 1. Thus, by Cauchy-Schwarz:
        //   -1 <= dotprod <= +1
        DEBUG_ASSERT(-1.0001 <= dotprod && dotprod <= 1.0001);
        output(oband, top, left) = dotprod;
      }
    } // end for all locations
  } // end for all filters
}

void CRbf(const ArrayRef3D<float>& input, const ArrayRef4D<float>& kernels,
    float beta, int scaling, ArrayRef3D<float>& output) {
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
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);

  // for all filters
  for (int oband = 0; oband < num_kernels; ++oband) {
    // for all locations
    for (int top = 0; top < oheight; ++top) {
      for (int left = 0; left < owidth; ++left) {
        int rj = scaling * top;
        int ri = scaling * left;
        float square_dist = 0.0;

        // for all kernel bands
        for (int iband = 0; iband < num_ibands; ++iband) {
          for (int kj = 0; kj < kheight; ++kj) {
            for (int ki = 0; ki < kwidth; ++ki) {
              float x = input(iband, rj + kj, ri + ki);
              float k = kernels(oband, iband, kj, ki);
              square_dist += (k - x) * (k - x);
            }
          }
        } // end for all kernel bands

        // Radial basis function with arbitrary vectors.
        float result = exp(-1 * beta * square_dist);

        output(oband, top, left) = result;
      }
    } // end for all locations
  } // end for all filters
}

void CNormRbf(const ArrayRef3D<float>& input,
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
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth);
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
        //   float input_norm = sqrt(squares + bias);
        // An alternative approach to avoid amplifying noise (used, e.g., by
        // Pinto et al) is to clip the divisor.
        squares = (squares < bias ? bias : squares);
        float input_norm = sqrt(squares);

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

// Pools over spatial location.
void CLocalMax(const ArrayRef3D<float>& input, int kheight,
    int kwidth, int scaling, ArrayRef3D<float>& output) {
  int num_bands = input.w0;
  ASSERT_EQUALS(num_bands, output.w0);
  int iheight = input.w1;
  int iwidth = input.w2;
  int max_oheight, max_owidth;
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth,
      &max_oheight, &max_owidth);
  int oheight = output.w1;
  int owidth = output.w2;
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);
  Index2D kshape(kheight, kwidth);
  for (int band = 0; band < num_bands; ++band) {
    for (int top = 0; top < oheight; ++top) {
      for (int left = 0; left < owidth; ++left) {
        int sj = scaling * top;
        int si = scaling * left;
        float max_activity = MINIMUM_NEGATIVE_FLOAT;
        for (int kj = 0; kj < kheight; ++kj) {
          for (int ki = 0; ki < kwidth; ++ki) {
            float activity = input(band, sj + kj, si + ki);
            if (activity > max_activity) {
              max_activity = activity;
            }
          }
        } // end processing single band and location
        output(band, top, left) = max_activity;
      }
    }
  } // end for bands and locations
}
