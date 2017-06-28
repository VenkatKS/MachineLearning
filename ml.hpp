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
private:
	bool is_normalized = false;
public:
	bool debug_print = false;

	ML_LinearOps(bool is_normalized)
	{
		this->is_normalized = is_normalized;
	}

	/* Compute the cost of the provided parameters for the provided data set */
	double computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);
	Matrix *gradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, double alpha, int num_iterations);

};

#endif /* ml_hpp */
