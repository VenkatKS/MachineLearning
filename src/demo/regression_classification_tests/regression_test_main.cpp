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
#include "log_test/log_test.hpp"

#define SET_REGRESSION_TEST_RUN 1

#if SET_REGRESSION_TEST_RUN
int main()
{
	printf("Machine Learning Library Regression Test\n");

	/* Test 2 and 3 variable linear learning */
	linear_regression_test *test_linear = new linear_regression_test();

	bool result = true;

	result = test_linear->run_test();
	if (!result)
	{
		printf("ERROR: Linear Regression regression test failed.\n");
		return (-1);
	}

	/* Test 2 and 3 variable Library Classification Test */
	log_regression_test *test_log = new log_regression_test();

	result = test_log->run_test();

	if (!result)
	{
		printf("ERROR: Log Classification regression test failed.");
	}
	
}
#endif
