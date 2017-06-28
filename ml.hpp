//
//  ml.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_hpp
#define ml_hpp

#include <stdio.h>
#include "2DMatrix.hpp"

/* Linear Regression Machine Learning Operations */
class ML_LinearOps
{
public:
	/* Compute the cost of the provided parameters for the provided data set */
	static double computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);

};

#endif /* ml_hpp */
