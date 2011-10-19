/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "util.h"

#include <string>
#include <sstream>
#include <cstdlib>
#include <execinfo.h>

MyException::MyException(const std::string& msg) : msg_(msg) {
}

MyException::MyException(const std::string& msg, const char* file, int line) {
  std::stringstream ss;
  ss << msg << " at " << file << ":" << line;
  msg_ = ss.str();
}

MyException::~MyException() throw() {
}

const char* MyException::what() const throw() {
  return msg_.c_str();
}

void handler(const std::string& msg, const char* file, int line) {
  const int max_frames = 20;
  void *frames[max_frames];
  int num_frames = backtrace(frames, max_frames);
  char** frame_strings = backtrace_symbols(frames, num_frames);
  std::stringstream buffer;
  buffer << msg << " at " << file << ":" << line << "\n";
  buffer << "Stack trace below:\n";
  for (int i = 1; i < num_frames; ++i)
    buffer << frame_strings[i] << "\n";
  free(frame_strings);
  throw MyException(buffer.str());
}

void COutputMapShapeForInput(int kheight, int kwidth, int scaling,
    int input_height, int input_width, int* output_height, int* output_width) {
  *output_height = (input_height - kheight + 1) / scaling;
  *output_width = (input_width - kwidth + 1) / scaling;
}

void CInputMapShapeForOutput(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width) {
  *input_height = output_height * scaling + kheight - 1;
  *input_width = output_width * scaling + kwidth - 1;
}

bool IsEnabledSSE() {
#ifdef __SSE__
  return true;
#else
  return false;
#endif
}

bool IsEnabledDebugging() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

void CMaxOutputDimensionsSSE(int kheight, int kwidth, int scaling, int input_height,
    int input_width, int* output_height, int* output_width, int* multi_kwidth,
    int* left_pad) {
  // Given a kernel of height multi_kheight, there are
  //   iwidth - multi_kheight + 1
  // possible vertical locations to apply the kernel to the input matrix. Make
  // sure we have enough elements in the output matrix.
  // *output_height = input_height / scaling - kheight + 1;
  *output_height = (input_height - kheight + 1) / scaling;

  int half_width = kwidth / 2;
  *left_pad = 0;
  // If kwidth is not divisible by number of vector elements, then pad on the
  // left to fix this.
  if (half_width % SSE_VECTOR_WIDTH) {
      *left_pad = SSE_VECTOR_WIDTH - (half_width % SSE_VECTOR_WIDTH);
  }
  // Total kernel width must accommodate one horizontal position per vector
  // element (see below). Thus, the minimum right-hand pad width is one less
  // than this.
  int right_pad = SSE_VECTOR_WIDTH - 1;
  *multi_kwidth = *left_pad + kwidth + right_pad;
  if (*multi_kwidth % SSE_VECTOR_WIDTH) {
      // Increase kernel width to make it an even multiple of the vector width,
      // leaving enough room for one horizontal position per vector element.
      right_pad = (2 * SSE_VECTOR_WIDTH - 1) - (*multi_kwidth %
          SSE_VECTOR_WIDTH);
      *multi_kwidth = *left_pad + kwidth + right_pad;
  }
  ASSERT_EQUALS(*multi_kwidth % SSE_VECTOR_WIDTH, 0);
  // The relation between input_width and output_width is the same, but
  // multi_kwidth is in units of 4 pixels.
  // *output_width = input_width / scaling - *multi_kwidth + SSE_VECTOR_WIDTH;
  // XXX need to verify that this doesn't make the output array too large...
  *output_width = (input_width - *multi_kwidth + SSE_VECTOR_WIDTH) / scaling;
}

void CInputDimensionsForOutputSSE(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width) {
  *input_height = output_height * scaling + kheight - 1;

  int half_width = kwidth / 2;
  int left_pad = 0;
  // If kwidth is not divisible by number of vector elements, then pad on the
  // left to fix this.
  if (half_width % SSE_VECTOR_WIDTH) {
      left_pad = SSE_VECTOR_WIDTH - (half_width % SSE_VECTOR_WIDTH);
  }
  // Total kernel width must accommodate one horizontal position per vector
  // element (see below). Thus, the minimum right-hand pad width is one less
  // than this.
  int right_pad = SSE_VECTOR_WIDTH - 1;
  int multi_kwidth = left_pad + kwidth + right_pad;
  if (multi_kwidth % SSE_VECTOR_WIDTH) {
      // Increase kernel width to make it an even multiple of the vector width,
      // leaving enough room for one horizontal position per vector element.
      right_pad = (2 * SSE_VECTOR_WIDTH - 1) - (multi_kwidth %
          SSE_VECTOR_WIDTH);
      multi_kwidth = left_pad + kwidth + right_pad;
  }
  ASSERT_EQUALS(multi_kwidth % SSE_VECTOR_WIDTH, 0);
  *input_width = output_width * scaling + multi_kwidth - SSE_VECTOR_WIDTH;
}
