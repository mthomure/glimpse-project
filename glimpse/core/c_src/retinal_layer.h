/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __RETINAL_LAYER_H__
#define __RETINAL_LAYER_H__

#include "array.h"

void CProcessRetina(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output);

void CProcessRetinaSSE(const ArrayRef2D<float>& input, int kheight, int kwidth,
    float bias, ArrayRef2D<float>& output);

#endif // __RETINA_H__

