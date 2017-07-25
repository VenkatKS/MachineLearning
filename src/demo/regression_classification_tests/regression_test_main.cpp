//
//  regression_test_main.cpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/11/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "regression_test_main.hpp"
#include "../../lib/OpenCL/include/opencl_driver.hpp"
#include "linear_test/linear_test.hpp"

#define SET_REGRESSION_TEST_RUN 1

#if SET_REGRESSION_TEST_RUN
int main()
{
	printf("Machine Learning Library Regression Test\n");

	setUpOpenCLDrivers();

	/* Test 2 and 3 variable linear learning */
	linear_regression_test *test_linear = new linear_regression_test();

	bool result = true;

	result = test_linear->run_test();
	if (!result)
	{
		printf("ERROR: Linear 2-variable regression failed.\n");
		return (-1);
	}

	
}
#endif
