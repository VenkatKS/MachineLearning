////////////////////////////////////////////////////////////////////////////////

#include "include/ml_opencl_kernels.hpp"
#include "include/opencl_driver.hpp"
#include "../LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../MachineLearningLibrary.hpp"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>


/* The current active OPENCL kernel configuration */
opencl_driver *active_driver = NULL;

int opencl_driver::setupKernel(int k_id, const char *kernName, const char *progName)
{
	int err;

	cl_device_id &device_id = all_kern[k_id].device_id;
	cl_context &context = all_kern[k_id].context;
	cl_command_queue &commands = all_kern[k_id].commands;
	cl_program &program = all_kern[k_id].program;
	cl_kernel &kernel = all_kern[k_id].kernel;

	// Connect to a compute device
	//
	int gpu = 1;
	err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	program = clCreateProgramWithSource(context, 1, (const char **) & kernName, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, progName, &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	return 0;
}

int opencl_driver::execute_kernel(ml_opencl_execution_state &execution_environment)
{
	int k_id = execution_environment.k_id;
	long mem_one = execution_environment.input_one_sz;
	long mem_two = execution_environment.input_two_sz;
	long mem_three = execution_environment.output_sz;

	unsigned int count1 = execution_environment.numArg1;
	unsigned int count2 = execution_environment.numArg2;
	unsigned int count3 = execution_environment.numArg3;

	float *input_one = execution_environment.input_one;
	float *input_two = execution_environment.input_two;
	float *out = execution_environment.output;

	size_t *global = execution_environment.global;

	cl_uint numDims = execution_environment.numDimms;

	cl_context &context = all_kern[k_id].context;
	cl_kernel &kernel = all_kern[k_id].kernel;
	cl_command_queue &commands = all_kern[k_id].commands;

	cl_mem input1 = nullptr;
	cl_mem input2 = nullptr;
	cl_mem output = nullptr;

	int err = 0;
	if (input_one)
		input1 = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * mem_one, input_one, &err);
	if (err != CL_SUCCESS)
		goto failed;
	if (input_two)
		input2 = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * mem_two, input_two, &err);
	if (err != CL_SUCCESS)
		goto failed;
	output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * mem_three, NULL, &err);

	if ((input_one && !input1) || (input_two && !input2) || !output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}
failed:
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write to source array!\n");
		exit(1);
	}

	// Set the arguments to our compute kernel
	//
	err = 0;
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count1);
	err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &count2);
	err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &count3);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	//
	err = clEnqueueNDRangeKernel(commands, kernel, numDims, NULL, global, NULL, 0, NULL, NULL);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}

	// Wait for the command commands to get serviced before reading back results
	//
	clFinish(commands);

	// Read back the results from the device to verify the output
	//
	err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * mem_three, out, 0, NULL, NULL );
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
	if (input_one)
		clReleaseMemObject(input1);
	if (input_two)
		clReleaseMemObject(input2);
	clReleaseMemObject(output);

	return 0;
}

int opencl_driver::destroy_kernel(int k_id)
{
	cl_context &context = all_kern[k_id].context;
	cl_kernel &kernel = all_kern[k_id].kernel;
	cl_command_queue &commands = all_kern[k_id].commands;
	cl_program &program = all_kern[k_id].program;

	// Shutdown and cleanup
	//
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}

int opencl_driver::setUpOpenCLDrivers()
{
	/* Set to main to exercise OPENCL driver changes */

	/* Set up matrix scalar power kernel */
	setupKernel(1, MatrixPowerScalar, "matrixpowerscalar");
	/* Set up matrix scalar multiply kernel */
	setupKernel(2, MatrixMultiply, "matrixMul");
	/* Set up Matrix + Matrix kernel */
	setupKernel(3, MatrixAddMatrix, "matrixadd");
	/* Set up Matrix - Matrix kernel */
	setupKernel(4, MatrixSubtractMatrix, "matrixsub");
	/* Set up Matrix ^ Matrix kernel */
	setupKernel(5, MatrixPowerMatrix, "matrixpower");
	/* Set up Matrix Mean kernel */
	setupKernel(6, MatrixMean, "matrixmean");
	/* Set up Matrix Sum kernel */
	setupKernel(7, MatrixSum, "matrixsum");
	/* Set up Matrix STD DEV kernel */
	setupKernel(8, MatrixStdDev, "matrixstddev");
	/* Set up Matrix transpose kernel */
	setupKernel(9, MatrixTranspose, "matrixtranspose");

	return 0;
}

