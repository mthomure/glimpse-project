/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __UTIL_H__
#define __UTIL_H__

#include <cfloat>
#include <sstream>
#include <assert.h>
#include <exception>

const float MINIMUM_NEGATIVE_FLOAT = -FLT_MAX;

/// Assertions ///

class MyException : public std::exception {
 private:
  std::string msg_;
 public:
  MyException(const std::string& msg);
  MyException(const std::string& msg, const char* file, int line);
  virtual ~MyException() throw();
  const char* what() const throw();
};

void handler(const std::string& msg, const char* file, int line);

#define __ASSERT_HANDLER(EXPR, MSG)   ((EXPR) ? ((void)0) : throw MyException((MSG), __FILE__, __LINE__))
#define ASSERT_TRUE(value)  __ASSERT_HANDLER((value), "ASSERT_TRUE")
#define ASSERT_EQUALS(A, B) __ASSERT_HANDLER(((A) == (B)), "ASSERT_EQUALS")
#define ASSERT_NOT_NULL     ASSERT_TRUE

template <class T1, class T2>
void ThrowEqualsException(const T1& actual, const T2& expected, const char* file, int line) {
  std::stringstream ss;
  ss << "EqualsException: expected " << expected << " but got " << actual << " at " << file << ":" << line;
  throw MyException(ss.str());
}

#ifdef NDEBUG
# define DEBUG_ASSERT(value)    ((void)0)
#else
# define DEBUG_ASSERT(value)    __ASSERT_HANDLER((value), "DEBUG_ASSERT")
#endif

bool IsEnabledDebugging();
void CMaxOutputDimensions(int kheight, int kwidth, int scaling,
    int input_height, int input_width, int* output_height, int* output_width);
void CInputDimensionsForOutput(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width);

#endif // __UTIL_H__
