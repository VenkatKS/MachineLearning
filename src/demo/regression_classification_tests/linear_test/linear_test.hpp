//
//  linear_regression_test.hpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/11/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef linear_regression_test_hpp
#define linear_regression_test_hpp

#include <stdio.h>
#include "../regression_test_main.hpp"
#include "../../../lib/LinearAlgebraLibrary/include/2DMatrix.hpp"

class linear_regression_test:public regression_test
{
private:
	bool run_single_feature_test();		/* 1 variable polynomial, and 1 bias test */
	bool run_multi_feature_test();	/* 2 variable polynomial, and 1 bias test */
public:
	bool run_test();
};

#endif /* linear_regression_test_hpp */
