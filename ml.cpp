//
//  ml.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "ml.hpp"


double ML_LinearOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
{
	int numTrainingExamples = training_y.numRows();
	double currentCost = 0;
	int idx = 0;

	Matrix *X = new Matrix(training_X);
	X->AddBiasCol();

	/* Calculate the hypothesis */
	Matrix *result = (*X) * training_theta;

	if (!result)
	{
		printf("ERROR: Matrix dimensions between parameters and training examples must agree.\n");
		return -1;
	}

	result = (*result) - training_y;

	if (!result)
	{
		printf("ERROR: Matrix dimensions for training set invalid.\n");
		return -1;
	}

	result->PowerScalar(2);

	double divisor = (1.0 / (2.0 * (double) numTrainingExamples));
	for (idx = 0; idx < numTrainingExamples; idx++)
	{
		currentCost += divisor * (*result)[idx];
	}

	return currentCost;
}

Matrix *ML_LinearOps::gradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, double alpha, int num_iterations)
{
	int numTrainingExamples = training_y.numRows();
	int iteration_idx = 0;
	double min_constant = alpha * (1/((double) numTrainingExamples));

	Matrix *result = new Matrix(theta);
	Matrix *temp_result;
	Matrix *hypothesis;
	Matrix *error;
	Matrix *X = new Matrix(training_X);
	X->AddBiasCol();
	Matrix *gradient;

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		hypothesis = (*X) * (*result);
		error = (*hypothesis) - training_y;
		X->Transpose();
		gradient = (*X) * (*error);
		X->Transpose();

		gradient->MultiplyScalar(min_constant);
		temp_result = (*result) - (*gradient);

		delete result;
		delete hypothesis;
		delete error;
		delete gradient;

		result = temp_result;
		printf("%f\n", computeCost(training_X, training_y, *result));

	}

	return result;
}
