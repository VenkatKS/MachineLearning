//
//  ml_opencl_kernels.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/5/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_opencl_kernels_hpp
#define ml_opencl_kernels_hpp

#include <stdio.h>

const char *MatrixSquareElements =					"\n" \
"__kernel void matrixsquareelements(					\n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count1,						\n" \
"   const unsigned int count2,						\n" \
"   const unsigned int count3)						\n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count3)                                                      \n" \
"       output[i] = input1[i] * input2[i];                              \n" \
"}                                                                      \n" \
"\n";

const char *MatrixPowerScalar =						"\n" \
"__kernel void matrixpowerscalar(					\n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count1,						\n" \
"   const unsigned int count2,						\n" \
"   const unsigned int count3)						\n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count3)                                                      \n" \
"       output[i] = pow(input1[i], input2[0]);                          \n" \
"}                                                                      \n" \
"\n";


const char *MatrixMultiply =
	"// OpenCL Kernel\n"
	"__kernel void \n"
	"matrixMul("
	"__global float* A, \n"
	"__global float* B, \n"
	"__global float* C, \n"
	"int wA, int wB, int wC)\n"
	"{\n"
		"int tx = get_global_id(0); \n" // op1_r_idx
		"int ty = get_global_id(1);\n" // op2_c_idx
		"if (tx >= wC) return; \n"
		"if (ty >= wB) return;"
		"float value = 0;\n"
		"for (int k = 0; k < wA; ++k)\n" // op1_c_idx
		"{\n"
			"float elementA = A[tx * wA + k];\n"
			"float elementB = B[k * wB + ty];\n"
			"value += elementA * elementB;\n"
		"}\n"
		"C[tx * wB + ty] = value;\n"
	"}\n";

const char *MatrixAddMatrix =						"\n"\
"__kernel void matrixadd(						\n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count1,						\n" \
"   const unsigned int count2,						\n" \
"   const unsigned int count3)						\n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count3)                                                      \n" \
"       output[i] = input1[i] + input2[i];				\n" \
"}                                                                      \n" \
"\n";

const char *MatrixSubtractMatrix =					"\n"\
"__kernel void matrixsub(						\n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count1,						\n" \
"   const unsigned int count2,						\n" \
"   const unsigned int count3)						\n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count3)                                                      \n" \
"       output[i] = input1[i] - input2[i];				\n" \
"}                                                                      \n" \
"\n";

const char *MatrixPowerMatrix =						"\n" \
"__kernel void matrixpower(						\n" \
"   __global float* input1,                                              \n" \
"   __global float* input2,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count1,						\n" \
"   const unsigned int count2,						\n" \
"   const unsigned int count3)						\n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count3)                                                       \n" \
"       output[i] = pow(input1[i], input2[i]);                          \n" \
"}                                                                      \n" \
"\n";

const char *MatrixMean =
"__kernel void matrixmean(							\n" \
"   __global float* input1,							\n" \
"   __global float* input2,							\n" \
"   __global float* output,							\n" \
"   const unsigned int numRows,							\n" \
"   const unsigned int numCols,							\n" \
"   const unsigned int count3)							\n" \
"{										\n" \
"	int r_idx = 0;								\n" \
"	int i = get_global_id(0);						\n" \
"	float value = 0;							\n" \
"	if(i < numCols) {							\n" \
"		for (r_idx = 0; r_idx < numRows; r_idx++)			\n" \
"		{								\n" \
"			value = value + input1[(r_idx * numCols) + i];		\n" \
"		}								\n" \
"	}									\n" \
"	output[i] = value / numRows;						\n" \
"}										\n" \
"\n";

const char *MatrixSum =
"__kernel void matrixsum(							\n" \
"   __global float* input1,							\n" \
"   __global float* input2,							\n" \
"   __global float* output,							\n" \
"   const unsigned int numRows,							\n" \
"   const unsigned int numCols,							\n" \
"   const unsigned int count3)							\n" \
"{										\n" \
"	int r_idx = 0;								\n" \
"	int i = get_global_id(0);						\n" \
"	float value = 0;							\n" \
"	if(i < numCols) {							\n" \
"		for (r_idx = 0; r_idx < numRows; r_idx++)			\n" \
"		{								\n" \
"			value = value + input1[(r_idx * numCols) + i];		\n" \
"		}								\n" \
"		output[i] = value;						\n" \
"	}									\n" \
"}										\n" \
"\n";

const char *MatrixStdDev =
"__kernel void matrixstddev(										\n" \
"   __global float* matrixvalues,									\n" \
"   __global float* matrixmeans,									\n" \
"   __global float* output,										\n" \
"   const unsigned int numRows,										\n" \
"   const unsigned int numCols,										\n" \
"   const unsigned int count3)										\n" \
"{													\n" \
"	int r_idx = 0;											\n" \
"	int i = get_global_id(0);									\n" \
"	float value = 0;										\n" \
"	if(i < numCols) {										\n" \
"		float colRunningCount = 0;								\n" \
"		float colMean = matrixmeans[i];								\n" \
"		for (r_idx = 0; r_idx < numRows; r_idx++)						\n" \
"		{											\n" \
"			float indexValue = matrixvalues[(r_idx * numCols) + i];				\n" \
"			indexValue = indexValue - colMean;						\n" \
"			indexValue = fabs(indexValue);							\n" \
"			indexValue = pow(indexValue, (float) 2);					\n" \
"			colRunningCount = colRunningCount + indexValue;					\n" \
"		}											\n" \
		/* NOTE: Uses MATLAB's formula for standard deviation */
"		colRunningCount = (((float)(colRunningCount)) / ((float) (numRows - 1)));		\n" \
"		output[i] = (sqrt(colRunningCount));							\n" \
"	}												\n" \
"}													\n" \
"\n";

const char *MatrixTranspose =
"__kernel void matrixtranspose(								\n" \
"   __global float* matrix,								\n" \
"   __global float* not_used,								\n" \
"   __global float* output,								\n" \
"   const unsigned int numRows,								\n" \
"   const unsigned int numCols,								\n" \
"   const unsigned int not_used_1)							\n" \
"{											\n" \
"	int r_idx = get_global_id(0);							\n" \
"	int c_idx = get_global_id(1);							\n" \
"	float value = 0;								\n" \
"	if(r_idx < numRows && c_idx < numCols) {					\n" \
"		output[(c_idx * numRows) + r_idx] = matrix[(r_idx * numCols) + c_idx];	\n" \
"	}										\n" \
"}											\n" \
"\n";


#endif /* ml_opencl_kernels_hpp */
