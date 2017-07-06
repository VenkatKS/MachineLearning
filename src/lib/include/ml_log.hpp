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
public:
	bool debug_print = false;


	/* Compute the cost of the provided parameters for the provided data set */
	static Matrix *sigmoid(Matrix &z);
	static float computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);
	static Matrix *gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta);
	static Matrix *GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations);
	static Matrix *Predict(Matrix &input_examples, Matrix &theta, float threshold);

	/* Regularized Operations */
	static float computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta, float regularizationParam);
	static Matrix *gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta, float regularizationParam);
	static Matrix *GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations, float regularizationParam);

	/* Multi-Class Classification */
	static Matrix *OneVsAll(Matrix &training_X, Matrix &training_y, int num_classes, double alpha, int num_iterations, double regularizationParam);
	static Matrix *PredictOneVsAll(Matrix &training_X, Matrix &all_theta);


};

#endif /* ml_log_hpp */
