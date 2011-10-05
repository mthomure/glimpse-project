/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __FILTERS_H__
#define __FILTERS_H__

#include "array.h"
#include "bitset_array.h"

void CContrastEnhance(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output);

void CDotProduct(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, int scaling, ArrayRef3D<float>& output);

void CNormDotProduct(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, int scaling,
    ArrayRef3D<float>& output);

void CNormRbf(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, float beta, int scaling,
    ArrayRef3D<float>& output);

void CRbf(const ArrayRef3D<float>& input, const ArrayRef4D<float>& kernels,
    float beta, int scaling, ArrayRef3D<float>& output);

void CLocalMax(const ArrayRef3D<float>& input, int kheight,
    int kwidth, int scaling, ArrayRef3D<float>& output);

#endif // __FILTERS_H__
