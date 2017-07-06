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
	bool debug_print = false;

	/* Compute the cost of the provided parameters for the provided data set */
	static float computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);
	static Matrix *gradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations);
	static Matrix *Predict(Matrix &predict_X, Matrix &theta);

	/* FIXME: Implement normal equation */
	static Matrix *normalEquation(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations);
};


class ML_DataOps
{
public:
	/* Return a normalized data set, where the mean is 0 and the std dev is 1 */
	static Matrix *NormalizeData(Matrix &data);
};
#endif /* ml_hpp */
