//
//  ml.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <math.h>
#include <cassert>
#include "../LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../MachineLearningLibrary.hpp"
#include "include/ml_regression.hpp"


float ML_LinearOps::computeCost(Matrix &parameters_to_evaluate)
{
	float currentCost = 0;
	int idx = 0;

	Matrix *X = (new Matrix(*this->data_x));
	X->AddBiasCol();

	/* Calculate the hypothesis */
	Matrix *result = (*X) * parameters_to_evaluate;

	if (!result)
	{
		printf("ERROR: Matrix dimensions between parameters and training examples must agree.\n");
		return -1;
	}

	Matrix *temp_result = (*result) - (*this->data_y);

	delete result;

	result = temp_result;

	if (!result)
	{
		printf("ERROR: Matrix dimensions for training set invalid.\n");
		return -1;
	}

	result->operateOnMatrixValues(2, OP_RAISE_EVERY_MATRIX_ELEMENT_TO_SCALAR_POWER);

	float divisor = (1.0 / (2.0 * (float) this->numTrainingExamples));
	for (idx = 0; idx < this->numTrainingExamples; idx++)
	{
		currentCost += divisor * (*result)[idx];
	}

	delete result;
	delete X;

	return currentCost;
}

Matrix *ML_LinearOps::gradientCalculate(Matrix &params_to_derivate)
{
	/* FIXME: Remove related features from gradientDescent into this function */
	return NULL;
}

Matrix *ML_LinearOps::Optimize(Matrix &initial_params,  int num_iterations)
{
	int iteration_idx = 0;
	float min_constant = this->alpha * (1/((float) this->numTrainingExamples));

	Matrix *result = new Matrix(initial_params);
	Matrix *temp_result;
	Matrix *hypothesis;
	Matrix *error;
	Matrix *X = new Matrix(*this->data_x);
	X->AddBiasCol();
	Matrix *gradient;

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		hypothesis = (*X) * (*result);
		error = (*hypothesis) - (*this->data_y);
		if (error == NULL)
		{
			printf("ERROR: Provided training solution dimensions do not agree with provided training set.\n");
			assert (0);
			return NULL;
		}

		X->Transpose();
		gradient = (*X) * (*error);
		X->Transpose();

		gradient->operateOnMatrixValues(min_constant, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);
		temp_result = (*result) - (*gradient);

		/* Clean up temporary matrices */
		delete result;
		delete hypothesis;
		delete error;
		delete gradient;

		result = temp_result;
	}

	/* FIXME: Add normalization warning?? */

	delete X;

	return result;
}

Matrix *ML_LinearOps::Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold)
{
	Matrix *X = new Matrix(input_to_evaluate);
	X->AddBiasCol();
	Matrix *result = (*X) * parameters;
	delete X;
	return result;
}
