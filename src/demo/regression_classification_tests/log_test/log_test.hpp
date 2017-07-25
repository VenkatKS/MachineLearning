//
//  log_test.hpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/17/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef log_test_hpp
#define log_test_hpp

#include <stdio.h>
#include "../regression_test_main.hpp"

class log_regression_test:public regression_test
{
private:
	bool run_single_feature_test();		/* 1 variable classification, and 1 bias test */
	bool run_multi_feature_test();		/* 2 variable classification, and 1 bias test */
public:
	bool run_test();
};

#endif /* log_test_hpp */
