//
//  ml_log.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/28/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_log_hpp
#define ml_log_hpp

#include <stdio.h>
#include "2DMatrix.hpp"

/* Linear Regression Machine Learning Operations */
class ML_LogOps
{
private:
	bool is_regularized = false;

public:
	bool debug_print = false;

	ML_LogOps(bool is_regularized)
	{
		this->is_regularized = is_regularized;
	}

	/* Compute the cost of the provided parameters for the provided data set */
	static Matrix *sigmoid(Matrix &z);
	static double computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);
	static Matrix *gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta);
	static Matrix *GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, double alpha, int num_iterations);
};

#endif /* ml_log_hpp */
