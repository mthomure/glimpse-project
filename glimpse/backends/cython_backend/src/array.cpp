/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "array.h"
#include <cmath>

Index2D::Index2D(int w0, int w1) : w0(w0), w1(w1) {
}

Index3D::Index3D(int w0, int w1, int w2) : w0(w0), w1(w1), w2(w2) {
}

Index4D::Index4D(int w0, int w1, int w2, int w3) : w0(w0), w1(w1), w2(w2),
    w3(w3) {
}

Index5D::Index5D(int w0, int w1, int w2, int w3, int w4) : w0(w0), w1(w1),
    w2(w2), w3(w3), w4(w4) {
}

// Normalize each location to have constant energy.
void CNormalizeArrayAcrossBand_UnitNorm(ArrayRef3D<float>* data) {
    int num_bands = data->w0;
    // Normalizing an array with a single band will zero the array,
    // since the value at each location is equal to the mean.
    if (num_bands < 2) {
        return;
    }
    int height = data->w1;
    int width = data->w2;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float sum = 0.0;
            for (int b = 0; b < num_bands; ++b) {
                sum += (*data)(b, j, i);
            }
            float mean = sum / num_bands;
            float centered_squares = 0;
            for (int b = 0; b < num_bands; b++) {
                float& v = (*data)(b, j, i);
                v -= mean;
                centered_squares += v * v;
            }
            float norm = sqrt(centered_squares);
            // Only scale if norm is big enough. This avoids amplifying noise.
            if (norm > 1.0) {
                for (int b = 0; b < num_bands; ++b) {
                    (*data)(b, j, i) /= norm;
                }
            }
        }
    }
}

