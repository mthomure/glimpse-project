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

void CMaxOutputDimensions(int kheight, int kwidth, int scaling, int input_height,
    int input_width, int* output_height, int* output_width) {
  // Given a kernel of height kheight, there are
  //   input_height - kheight + 1
  // possible vertical locations to apply the kernel to the input matrix. The
  // situation is analogous for the output width. Use this to compute the
  // bounding box of defined elements in the output layer.
  *output_height = (input_height - kheight + 1) / scaling;
  *output_width = (input_width - kwidth + 1) / scaling;
}

void CInputDimensionsForOutput(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width) {
  *input_height = output_height * scaling + kheight - 1;
  *input_width = output_width * scaling + kwidth - 1;
}

bool IsEnabledDebugging() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}
