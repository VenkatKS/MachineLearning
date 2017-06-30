//
//  ml.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <math.h>
#include "include/ml_linear.hpp"


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

	Matrix *temp_result = (*result) - training_y;

	delete result;

	result = temp_result;

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

	delete result;
	delete X;

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

	/* Row Vectors for storing the mean and std. dev. for each training feature across all examples */
	Matrix &Data_Mean = *(new Matrix(1, data.numCols()));
	Matrix &Data_STD = *(new Matrix(1, data.numCols()));

	/* Calculate the mean of each column (i.e. each feature) */
	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		double colRunningCount = 0;

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			colRunningCount = colRunningCount + data[currentIndex];
			delete currentIndex;
		}

		Indexer *currentMean = new Indexer(0, c_idx);
		Data_Mean[currentMean] = (((double)(colRunningCount)) / ((double) data.numRows()));
	}

	/* Calculate the std. dev. of each column (i.e. each feature) */
	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		Indexer *colMeanIndex = new Indexer(0, c_idx);
		double colRunningCount = 0;
		double colMean = Data_Mean[colMeanIndex];

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);

			double indexValue = data[currentIndex];
			indexValue = indexValue - colMean;
			indexValue = fabs(indexValue);
			indexValue = pow(indexValue, (double) 2);
			colRunningCount = colRunningCount + indexValue;

			delete currentIndex;
		}

		/* NOTE: Uses MATLAB's formula for standard deviation */
		colRunningCount = (((double)(colRunningCount)) / ((double) (data.numRows() - 1)));
		Data_STD[colMeanIndex] = (sqrt(colRunningCount));

		delete colMeanIndex;
	}

	/* Normalize the array */
	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		Indexer *colMeanIndex = new Indexer(0, c_idx);
		double colMean = Data_Mean[colMeanIndex];
		double colStd = Data_STD[colMeanIndex];

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);

			double indexValue = data[currentIndex];
			indexValue = indexValue - colMean;
			indexValue = indexValue / colStd;

			Data_Normalized[currentIndex] = indexValue;
			delete currentIndex;
		}
	}
	

	return &Data_Normalized;
}
