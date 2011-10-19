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

#ifdef __SSE__
#include <xmmintrin.h>

// Width of vector in bytes
#define SSE_VECTOR_WIDTH_BYTES 16
// Width of vector in (8-bit) words
#define SSE_VECTOR_WIDTH 4

/** Vector of four, single-precision floats. */
typedef __m128 v4f;

/** Swizzle the data from (xxxx, yyyy) format to (xyxy, xyxy) format and add.
 * Set x0123 = [ sum(data[0.1-0.3]), ..., sum(data[3.0-3.3]) ]
 */
inline v4f SwizzleAdd(v4f* data) {
  v4f x02   = _mm_add_ps( _mm_unpacklo_ps( data[0], data[2] ), _mm_unpackhi_ps( data[0], data[2] ) );
  v4f x13   = _mm_add_ps( _mm_unpacklo_ps( data[1], data[3] ), _mm_unpackhi_ps( data[1], data[3] ) );
  v4f x0123 = _mm_add_ps( _mm_unpacklo_ps(  x02,  x13 ), _mm_unpackhi_ps(  x02,  x13 ) );
  return x0123;
}
#endif  // __SSE__

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
//~ #define __ASSERT_HANDLER(EXPR, MSG)   ((EXPR) ? ((void)0) : handler((MSG), __FILE__, __LINE__))
//~ #define __ASSERT_HANDLER(EXPR, MSG)   assert(EXPR)

#define ASSERT_TRUE(value)  __ASSERT_HANDLER((value), "ASSERT_TRUE")
#define ASSERT_EQUALS(A, B) __ASSERT_HANDLER(((A) == (B)), "ASSERT_EQUALS")   //(((A) == (B)) ? ((void)0) : throw MyException("ASSERT_EQUALS", __FILE__, __LINE__)) //ThrowEqualsException(#A, #B, __FILE__, __LINE__))
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
//~ # define DEBUG_ASSERT(value)    ((value)      ? ((void)0) : handler("DEBUG_ASSERT", __FILE__, __LINE__))
# define DEBUG_ASSERT(value)    __ASSERT_HANDLER((value), "DEBUG_ASSERT")
#endif

bool IsEnabledSSE();
bool IsEnabledDebugging();
void CMaxOutputDimensionsSSE(int kheight, int kwidth, int scaling,
    int input_height, int input_width, int* output_height, int* output_width,
    int* multi_kwidth, int* lpad);
void CInputDimensionsForOutputSSE(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width);

void COutputMapShapeForInput(int kheight, int kwidth, int scaling,
    int input_height, int input_width, int* output_height, int* output_width);
void CInputMapShapeForOutput(int kheight, int kwidth, int scaling,
    int output_height, int output_width, int* input_height, int* input_width);

#endif // __UTIL_H__

