/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "complex_layer.h"
#include "util.h"

#include <iostream>
using std::cout;

// A C1 complex layer, which pools over phase and spatial location. This is
// specific to C1, because it assumes the shape of the input layer. From a
// generic standpoint, this pools over the second dimension of the input bands,
// and locally over the third and fourth dimensions.
void CProcessC1Layer_PoolSpaceAndPhase(const ArrayRef4D<float>& input,
    int kheight, int kwidth, int scaling, ArrayRef3D<float>& output,
    BitsetArrayRef3D& max_coords) {
  int num_orientations = input.w0;
  ASSERT_EQUALS(num_orientations, output.w0);
  int num_phases = input.w1;
  int num_bits_per_set = num_phases * kheight * kwidth;
  ASSERT_TRUE(num_bits_per_set <= max_coords.Get().num_bits_per_set);
  int iheight = input.w2;
  int iwidth = input.w3;
  int oheight = output.w1;
  int owidth = output.w2;
  int max_oheight, max_owidth;
  CMaxOutputDimensions(kheight, kwidth, scaling, iheight, iwidth, &max_oheight,
      &max_owidth);
  ASSERT_TRUE(oheight <= max_oheight);
  ASSERT_TRUE(owidth <= max_owidth);
  Index3D kshape(num_phases, kheight, kwidth);
  for (int theta = 0; theta < num_orientations; ++theta) {
    for (int top = 0; top < oheight; ++top) {
      for (int left = 0; left < owidth; ++left) {
        int sj = scaling * top;
        int si = scaling * left;
        float max_activity = MINIMUM_NEGATIVE_FLOAT;
        BitsetRef max_coord = max_coords(theta, top, left);
        for (int phi = 0; phi < num_phases; ++phi) {
          for (int kj = 0; kj < kheight; ++kj) {
            for (int ki = 0; ki < kwidth; ++ki) {
              float activity = input(theta, phi, sj + kj, si + ki);
              if (activity > max_activity) {
                max_activity = activity;
                max_coord.Clear();
                max_coord.Set( kshape(phi, kj, ki) );
              } else if (activity == max_activity) {
                max_coord.Set( kshape(phi, kj, ki) );
              }
            }
          }
        } // end processing single band and location
        output(theta, top, left) = max_activity;
      }
    }
  } // end for bands and locations
}

// A general complex layer, which pools over spatial location.
void CProcessComplexLayer(const ArrayRef3D<float>& input, int kheight,
    int kwidth, int scaling, ArrayRef3D<float>& output,
    BitsetArrayRef3D& max_coords) {
  int num_bands = input.w0;
  ASSERT_EQUALS(num_bands, output.w0);
  int iheight = input.w1;
  int iwidth = input.w2;
  int max_oheight, max_owidth;
  CMaxOutputDimensions(kheight, kwidth, scaling, iheight, iwidth, &max_oheight,
      &max_owidth);
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
        BitsetRef max_coord = max_coords(band, top, left);
        for (int kj = 0; kj < kheight; ++kj) {
          for (int ki = 0; ki < kwidth; ++ki) {
            float activity = input(band, sj + kj, si + ki);
            if (activity > max_activity) {
              max_activity = activity;
              max_coord.Clear();
              max_coord.Set( kshape(kj, ki) );
            } else if (activity == max_activity) {
              max_coord.Set( kshape(kj, ki) );
            }
          }
        } // end processing single band and location
        output(band, top, left) = max_activity;
      }
    }
  } // end for bands and locations
}

