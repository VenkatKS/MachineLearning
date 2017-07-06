//
//  ml.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <math.h>
#include "include/ml_linear.hpp"


float ML_LinearOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
{
	int numTrainingExamples = training_y.numRows();
	float currentCost = 0;
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

	Matrix *temp_result = (*result) - training_y;

	delete result;

	result = temp_result;

	if (!result)
	{
		printf("ERROR: Matrix dimensions for training set invalid.\n");
		return -1;
	}

	result->operateOnMatrixValues(2, OP_RAISE_EVERY_MATRIX_ELEMENT_TO_SCALAR_POWER);

	float divisor = (1.0 / (2.0 * (float) numTrainingExamples));
	for (idx = 0; idx < numTrainingExamples; idx++)
	{
		currentCost += divisor * (*result)[idx];
	}

	delete result;
	delete X;

	return currentCost;
}

Matrix *ML_LinearOps::gradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations)
{
	int numTrainingExamples = training_y.numRows();
	int iteration_idx = 0;
	float min_constant = alpha * (1/((float) numTrainingExamples));

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

Matrix *ML_LinearOps::Predict(Matrix &predict_X, Matrix &theta)
{
	Matrix *X = new Matrix(predict_X);
	X->AddBiasCol();
	Matrix *result = (*X) * theta;
	return result;
}

/*
 *	Assumptions:
 *		Matrix data is a dataset where the columns contain parameters/features and
 *		the rows contain the various training examples.
 */
Matrix *ML_DataOps::NormalizeData(Matrix &data)
{
	Matrix &Data_Normalized = *(new Matrix(data));
	int r_idx = 0;
	int c_idx = 0;

	/* Calculate the mean of each column (i.e. each feature) */
	Matrix &Data_Mean = (*data.Mean());
	/* Calculate the std. dev. of each column (i.e. each feature) */
	Matrix &Data_STD = *(data.StdDev());

	/* Normalize the array */
	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		Indexer *colMeanIndex = new Indexer(0, c_idx);
		float colMean = Data_Mean[colMeanIndex];
		float colStd = Data_STD[colMeanIndex];

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);

			float indexValue = data[currentIndex];
			indexValue = indexValue - colMean;
			indexValue = indexValue / colStd;

			Data_Normalized[currentIndex] = indexValue;
			delete currentIndex;
		}
		delete colMeanIndex;
	}
	
	delete &Data_Mean;
	delete &Data_STD;

	return &Data_Normalized;
}

