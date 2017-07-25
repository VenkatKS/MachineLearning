//
//  regression_test_main.hpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/11/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef regression_test_main_hpp
#define regression_test_main_hpp

#include <stdio.h>
#include <math.h>
#include "../../lib/MachineLearningLibrary.hpp"
#define ROUGHLY_EQUAL(_float_1, _float_2)	\
((fabs((float) ((_float_1 - _float_2)/(_float_2))) * 100) < ((float) EPSILON))


class regression_test
{
	/* In percent */
	float tolerance = 0.001;
public:
	void change_tolerance(float new_tolerance)
	{
		if (new_tolerance < 0.001) printf("Warning! Decreasing tolerance. Might affect results.\n");
		this->tolerance = new_tolerance;
	}

	inline bool roughly_equal(float value1, float value2)
	{
		return (((fabs((float) ((value1 - value2)/(value2))) * 100) < ((float) this->tolerance)));
	}

	virtual bool run_test() = 0;
};

#endif /* regression_test_main_hpp */
