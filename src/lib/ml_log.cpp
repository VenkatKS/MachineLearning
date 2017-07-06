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
#include <thread>

#include "include/ml_log.hpp"

#define DEBUGGING 0

/* Performs the sigmoid function on each item in the provided matrix */
Matrix *ML_LogOps::sigmoid(Matrix &z)
{
	Matrix &Sigmoid_Matrix = *(new Matrix(z));
	Sigmoid_Matrix.operateOnMatrixValues(-1.0, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	float exponent = exp((float) 1.0);
	Sigmoid_Matrix.operateOnMatrixValues(exponent, OP_RAISE_SCALAR_TO_EVERY_MATRIX_ELEMENT_POWER);
	Sigmoid_Matrix.operateOnMatrixValues(1, OP_ADD_SCALAR_TO_EVERY_MATRIX_ELEMENT);
	Sigmoid_Matrix.operateOnMatrixValues(1.0, OP_INVERT_EVERY_MATRIX_ELEMENT_AND_MULTIPLY_SCALAR);

	return &Sigmoid_Matrix;
}

/* Assumptions: Features in columns, examples in rows */
float ML_LogOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
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

	float *A = result.getRaw();
	float *res = sum_result.getRaw();
	const int result_cols = result.numCols();

	for (c_idx = 0; c_idx < result.numCols(); c_idx++)
	{
		float runningColCount = 0;
		for (r_idx = 0; r_idx < result.numRows(); r_idx++)
		{
			runningColCount = runningColCount + A[(r_idx * result_cols) + c_idx];
		}
		res[c_idx] = runningColCount;
	}

	sum_result.operateOnMatrixValues((float) ((float)1.0 / (float) numTrainingExamples), OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

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

	float final_cost = sum_result[0];
	delete &sum_result;

	return final_cost;
}

Matrix *ML_LogOps::gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta)
{
	/* 1/m * sum((hyp - y) .* X) */
	int numTrainingSet = training_X.numRows();
	Matrix &X = *(new Matrix(training_X));
	X.AddBiasCol();
	float constant = ((float) 1.0) / ((float) numTrainingSet);

	Matrix *interim_hypothesis = (X * theta);
	Matrix *hypothesis = sigmoid(*interim_hypothesis);
	Matrix *TermOne = (*hypothesis) - training_y;
	Matrix *TermTwo = NULL;
	Matrix *gradient = NULL;

	delete hypothesis;
	delete interim_hypothesis;

	assert(TermOne->numCols() == 1);
	TermOne->Transpose();
	TermTwo = (*TermOne) * X;
	TermOne->Transpose();
	gradient = TermTwo->Sum();
	assert (gradient->numCols() == theta.numRows());

	gradient->operateOnMatrixValues(constant, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	gradient->Transpose();

	delete TermOne;
	delete TermTwo;
	delete &X;

	return gradient;
}

Matrix *ML_LogOps::GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations)
{
	/* myTheta = myTheta - ((alpha/m) * X' * (sigmoid(X * myTheta) - y)); */

	int iteration_idx = 0;
	Matrix *result = new Matrix(theta);
	Matrix *X = new Matrix(training_X);
	X->AddBiasCol();
#if DEBUGGING
	float prevCost = 0;
#endif

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		Matrix *nextGradient = ML_LogOps::gradientCalculate(training_X, training_y, *result);
#if DEBUGGING
		float currentCost = ML_LogOps::computeCost(training_X, training_y, theta);
		assert (!((iteration_idx > 0) && (prevCost < currentCost)));
		prevCost = currentCost;
#endif
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

Matrix *ML_LogOps::Predict(Matrix &input_examples, Matrix &theta, float threshold)
{
	input_examples.AddBiasCol();
	Matrix *interim_hypothesis = input_examples * theta;
	Matrix *predictions = sigmoid(*(interim_hypothesis));
	predictions->operateOnMatrixValues(threshold, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GEQ_SCALAR);

	delete interim_hypothesis;
	return predictions;
}


/* Regularization Operations */
float ML_LogOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta, float regularizationParam)
{
	float unregularizedCost = ML_LogOps::computeCost(training_X, training_y, training_theta);
	int numTraining = training_X.numRows();
	float regFactor = (float) ((float) regularizationParam) / ((float) (2 * numTraining));
	Matrix &temp_theta = (*new Matrix(training_theta));

	/* Do not regularize the first parameter */
	assert (temp_theta.numCols() == 1);
	temp_theta[0] = 0;


	temp_theta.operateOnMatrixValues(2, OP_RAISE_EVERY_MATRIX_ELEMENT_TO_SCALAR_POWER);
	temp_theta.operateOnMatrixValues(regFactor, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	Matrix *sum = temp_theta.Sum();

	assert (sum->numRows() == 1 && sum->numCols() == 1);

	float regularizedSum = (*sum)[0];

	delete sum;
	delete &temp_theta;

	return (unregularizedCost + regularizedSum);
}

Matrix *ML_LogOps::gradientCalculate(Matrix &training_X, Matrix &training_y, Matrix &theta, float regularizationParam)
{
	Matrix &unregularizedGradients = (*ML_LogOps::gradientCalculate(training_X, training_y, theta));
	Matrix &temp_theta = (*new Matrix(theta));
	int numTraining = training_X.numRows();
	float regFactor = (float) ((float) regularizationParam) / ((float) (numTraining));

	/* Do not regularize the first parameter */
	temp_theta[0] = 0;


	temp_theta.operateOnMatrixValues(regFactor, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	Matrix *regularizedGradients = unregularizedGradients + temp_theta;

	delete &unregularizedGradients;
	delete &temp_theta;

	return regularizedGradients;
}

Matrix *ML_LogOps::GradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, float alpha, int num_iterations, float regularizationParam)
{
	/* myTheta = myTheta - ((alpha/m) * X' * (sigmoid(X * myTheta) - y)); */

	int iteration_idx = 0;
	Matrix *result = new Matrix(theta);
	Matrix *X = new Matrix(training_X);
	X->AddBiasCol();
#if DEBUGGING
	float prevCost = 0;
#endif

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		Matrix *nextGradient = ML_LogOps::gradientCalculate(training_X, training_y, *result, regularizationParam);
#if DEBUGGING
		/* When debugging, verify that gradient descent is actually decreasing/not increasing the cost */
		float currentCost = ML_LogOps::computeCost(training_X, training_y, theta, regularizationParam);
		assert (!((iteration_idx > 0) && (prevCost < currentCost)));
		prevCost = currentCost;
#endif
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

/* Class IDs are in order */
Matrix *ML_LogOps::OneVsAll(Matrix &training_X, Matrix &training_y, int num_classes, double alpha, int num_iterations, double regularizationParam)
{
	int class_idx = 0;
	int feature_idx = 0;
	Matrix **all_theta = new Matrix*[num_classes];

	/*
	 * Return matrix has the same number of cols as number of classes to represent the optimal theta for each class to fit
	 * the given data, and the same number of columns as the training set + 1 since we have the same number of features as
	 * the original data set plus 1 for the bias feature.
	 */
	Matrix &return_matrix = (*new Matrix(training_X.numCols() + 1, num_classes));

	for (class_idx = 0; class_idx < num_classes; class_idx++)
	{
		Matrix *temp_y = new Matrix(training_y);
		Matrix *currentTheta = new Matrix(training_X.numCols() + 1, 1);
		temp_y->operateOnMatrixValues(class_idx, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
		Matrix *theta_idx = ML_LogOps::GradientDescent(training_X, *temp_y, *currentTheta, alpha, num_iterations, regularizationParam);
		all_theta[class_idx] = theta_idx;
		delete temp_y;
		delete currentTheta;
	}

	for (class_idx = 0; class_idx < num_classes; class_idx++)
	{
		for (feature_idx = 0; feature_idx < (training_X.numCols() + 1); feature_idx++)
		{
			Indexer *currentFeatureForCurrentTheta = new Indexer(feature_idx, class_idx);
			return_matrix[currentFeatureForCurrentTheta] = (*all_theta[class_idx])[feature_idx];
			delete currentFeatureForCurrentTheta;
		}
	}

	delete[] all_theta;

	return &return_matrix;
}

Matrix *ML_LogOps::PredictOneVsAll(Matrix &training_X, Matrix &all_theta)
{
	Matrix &X = (*new Matrix(training_X));
	X.AddBiasCol();
	Matrix *hypothesis = X * all_theta;
	Matrix *predict = sigmoid(*hypothesis);
	predict->Transpose();
	Matrix *end = predict->MaxRowNumber();
	return end;
	
}
