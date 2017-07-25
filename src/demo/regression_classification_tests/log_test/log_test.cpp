//
//  log_test.cpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/17/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "log_test.hpp"
#include <cassert>

bool log_regression_test::run_single_feature_test()
{
	
	return true;
}

bool log_regression_test::run_multi_feature_test()
{
	return true;
}

bool log_regression_test::run_test()
{
	/* Run 2 variable linear test -- 1 feature, 1 bias */
	bool result = true;

	result = this->run_single_feature_test();

	if (result)
		printf("Single-Feature Log Regression Test Pass.\n");
	else
		assert (0);

	/* Run 3 variable linear test -- 2 features, 1 bias */
	result = this->run_multi_feature_test();

	if (result)
		printf("Multi-Feature Log Regression Test Pass.\n");
	else
		assert (0);

	return result;
}
