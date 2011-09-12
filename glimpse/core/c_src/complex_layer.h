/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __COMPLEX_LAYER_H__
#define __COMPLEX_LAYER_H__

#include "array.h"
#include "bitset_array.h"

void CProcessC1Layer_PoolSpaceAndPhase(const ArrayRef4D<float>& input,
    int kheight, int kwidth, int scaling, ArrayRef3D<float>& output,
    BitsetArrayRef3D& max_coords);

void CProcessComplexLayer(const ArrayRef3D<float>& input, int kheight,
    int kwidth, int scaling, ArrayRef3D<float>& output,
    BitsetArrayRef3D& max_coords);

#endif // __COMPLEX_LAYER_H__
