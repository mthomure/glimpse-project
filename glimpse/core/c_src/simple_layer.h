/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __SIMPLE_LAYER_H__
#define __SIMPLE_LAYER_H__

#include "array.h"

void CProcessSimpleLayer(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, float beta, int scaling,
    ArrayRef3D<float>& output);

void CProcessSimpleLayerSSE_NoScaling(const ArrayRef3D<float>& input,
    const ArrayRef4D<float>& kernels, float bias, float beta,
    ArrayRef3D<float>& output);

#endif // __SIMPLE_LAYER_H__

