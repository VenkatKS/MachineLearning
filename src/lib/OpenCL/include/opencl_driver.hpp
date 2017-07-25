//
//  opencl_driver.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/5/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef opencl_driver_hpp
#define opencl_driver_hpp

#include <stdio.h>
#include <OpenCL/opencl.h>


class ml_opencl_kernel_data
{
public:
	size_t global;                      // global domain size for our calculation
	size_t local;                       // local domain size for our calculation

	cl_device_id device_id;             // compute device id
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;		    // compute kernel
};

class ml_opencl_execution_state
{
public:
	int k_id;
	float *input_one;
	long input_one_sz; /* In number of floats, NOT in bytes */
	float *input_two;
	long input_two_sz; /* In number of floats, NOT in bytes */
	int numArg1;
	int numArg2;
	int numArg3;
	float *output;
	long output_sz; /* In number of floats, NOT in bytes */
	size_t *local;
	size_t *global;
	cl_uint numDimms;
};

int setUpOpenCLDrivers();
int execute_kernel(ml_opencl_execution_state &execution_environment);




#endif /* opencl_driver_hpp */
