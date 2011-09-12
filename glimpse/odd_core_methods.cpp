#include "odd_core_methods.h"

void UpdateCovariance(ArrayRef4D<float>& cov, ArrayRef3D<float>& idata, 
    ArrayRef1D<float>& means, int step) {

  for (int fb = 0; fb < nbands; ++fb) {
    for (int fy = hw; fy < height - hw; ++fy) {
      for (int fx = hw; fx < width - hw; ++fx) {
        float center_activity = idata(fb, fy, fx) - means(fb);
        for (int b = 0; b < nbands; ++b) {
          for (int y = 0; y < kw; ++y) {
            for (int x = 0; x < kw; ++x) {
              float lateral_activity = idata(b, fy + y - hw, fx + x - hw) 
                  - means(b);
              cov(fb, b, y, x) += center_activity * lateral_activity;
            }
          }
        }
      }
    }
  }
}

