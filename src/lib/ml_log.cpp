//
//  ml_log.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/28/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <math.h>
#include <cassert>
#include <complex>
#include "include/ml_log.hpp"

/* Performs the sigmoid function on each item in the provided matrix */
Matrix *ML_LogOps::sigmoid(Matrix &z)
{
	Matrix &Sigmoid_Matrix = *(new Matrix(z));
	Sigmoid_Matrix.operateOnMatrixValues(-1.0, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	double exponent = exp((double) 1.0);
	Sigmoid_Matrix.operateOnMatrixValues(exponent, OP_RAISE_SCALAR_TO_EVERY_MATRIX_ELEMENT_POWER);
	Sigmoid_Matrix.operateOnMatrixValues(1, OP_ADD_SCALAR_TO_EVERY_MATRIX_ELEMENT);
	Sigmoid_Matrix.operateOnMatrixValues(1.0, OP_INVERT_EVERY_MATRIX_ELEMENT_AND_MULTIPLY_SCALAR);

	return &Sigmoid_Matrix;
}

/* Assumptions: Features in columns, examples in rows */
double ML_LogOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
{
	/* cost = (1/m) * sum((-1 * y) .* log(hyp) - ((1-y) .* log(1 - hyp))); */

	int numTrainingExamples = training_y.numRows();
	int c_idx = 0;
	int r_idx = 0;

	Matrix &X = *(new Matrix(training_X));
	X.AddBiasCol();

	Matrix *interim_hypothesis = (X * training_theta);
	Matrix *hypothesis = sigmoid(*interim_hypothesis);

	Matrix &scale = *(new Matrix(training_y));
	scale.operateOnMatrixValues(-1, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	Matrix &scale_2 = *(new Matrix(training_y));
	scale_2.operateOnMatrixValues(1, OP_SUBTRACT_EVERY_MATRIX_ELEMENT_FROM_SCALAR);

	Matrix &hypothesis2 = *(new Matrix(*hypothesis));

	hypothesis->Log_e();

	hypothesis2.operateOnMatrixValues(1, OP_SUBTRACT_EVERY_MATRIX_ELEMENT_FROM_SCALAR);
	hypothesis2.Log_e();

	scale.Transpose();
	scale_2.Transpose();

	Matrix *result_1 = scale * (*hypothesis);

	Matrix *result_2 = scale_2 * (hypothesis2);

	Matrix &result = *((*result_1) - (*result_2));

	Matrix &sum_result = *(new Matrix(1, result.numCols()));
	for (c_idx = 0; c_idx < result.numCols(); c_idx++)
	{
		double runningColCount = 0;
		for (r_idx = 0; r_idx < result.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			runningColCount = runningColCount + result[currentIndex];
			delete currentIndex;
		}
		Indexer *currentMean = new Indexer(0, c_idx);
		sum_result[currentMean] = runningColCount;
		delete currentMean;
	}

	sum_result.operateOnMatrixValues((double) ((double)1.0 / (double) numTrainingExamples), OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	assert(sum_result.numCols() == 1 && sum_result.numRows() == 1);

	/* FIXME: Add Regularization Expression */

	delete &X;
	delete &hypothesis2;
	delete hypothesis;
	delete interim_hypothesis;
	delete &scale_2;
	delete &scale;
	delete result_1;
	delete result_2;
	delete &result;

	double final_cost = sum_result[0];
	delete &sum_result;

	return final_cost;
}

Matrix *ML_LogOps::gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta)
{
	/* 1/m * sum((hyp - y) .* X) */
	int numTrainingSet = training_X.numRows();
	Matrix &X = *(new Matrix(training_X));
	X.AddBiasCol();
	int c_idx, r_idx = 0;
	double constant = ((double) 1.0) / ((double) numTrainingSet);

	Matrix *interim_hypothesis = (X * theta);
	Matrix *hypothesis = sigmoid(*interim_hypothesis);
	Matrix *TermOne = (*hypothesis) - training_y;
	Matrix *TermTwo = new Matrix(X.numRows(), X.numCols());
	Matrix *gradient = new Matrix(1, theta.numRows());

	delete hypothesis;
	delete interim_hypothesis;

	assert(TermOne->numCols() == 1);
	for (r_idx = 0; r_idx < X.numRows(); r_idx++)
	{
		for (c_idx = 0; c_idx < X.numCols(); c_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			(*TermTwo)[currentIndex] = (*TermOne)[r_idx] * X[currentIndex];
			delete currentIndex;
		}
	}

	for (c_idx = 0; c_idx < TermTwo->numCols(); c_idx++)
	{
		double runningColCount = 0;
		for (r_idx = 0; r_idx < TermTwo->numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			runningColCount = runningColCount + (*TermTwo)[currentIndex];
			delete currentIndex;
		}
		Indexer *currentMean = new Indexer(0, c_idx);
		(*gradient)[currentMean] = runningColCount;
		delete currentMean;
	}

	gradient->operateOnMatrixValues(constant, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	gradient->Transpose();

	delete TermOne;
	delete TermTwo;
	delete &X;

	return gradient;
}

Matrix *ML_LogOps::GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, double alpha, int num_iterations)
{
	/* myTheta = myTheta - ((alpha/m) * X' * (sigmoid(X * myTheta) - y)); */

	int iteration_idx = 0;
	Matrix *result = new Matrix(theta);
	Matrix *X = new Matrix(training_X);
	X->AddBiasCol();

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		Matrix *nextGradient = ML_LogOps::gradientCalculate(training_X, training_y, *result);
		nextGradient->operateOnMatrixValues(alpha, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

		Matrix *temp_result;
		temp_result = (*result) - (*nextGradient);

		delete nextGradient;
		delete result;
		result = temp_result;
	}

	delete X;
	return result;
}
