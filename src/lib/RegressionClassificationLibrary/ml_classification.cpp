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
#include "../LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../MachineLearningLibrary.hpp"
#include "include/ml_classification.hpp"

#define DEBUGGING 0

Matrix *ML_SingleLogOps::sigmoid(Matrix &z)
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
float ML_SingleLogOps::computeCostInternal(Matrix &parametes_to_evaluate)
{
	/* cost = (1/m) * sum((-1 * y) .* log(hyp) - ((1-y) .* log(1 - hyp))); */
	int c_idx = 0;
	int r_idx = 0;

	Matrix &X = *(new Matrix(*this->data_x));
	X.AddBiasCol();

	Matrix *interim_hypothesis = (X * parametes_to_evaluate);
	Matrix *hypothesis = sigmoid(*interim_hypothesis);

	Matrix &scale = *(new Matrix(*this->data_y));
	scale.operateOnMatrixValues(-1, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	Matrix &scale_2 = *(new Matrix(*this->data_y));
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

Matrix *ML_SingleLogOps::gradientCalculateInternal(Matrix &params_to_derivate)
{
	/* 1/m * sum((hyp - y) .* X) */
	Matrix &X = *(new Matrix(*this->data_x));
	X.AddBiasCol();
	float constant = ((float) 1.0) / ((float) numTrainingExamples);

	Matrix *interim_hypothesis = (X * params_to_derivate);
	Matrix *hypothesis = sigmoid(*interim_hypothesis);
	Matrix *TermOne = (*hypothesis) - (*this->data_y);
	Matrix *TermTwo = NULL;
	Matrix *gradient = NULL;

	delete hypothesis;
	delete interim_hypothesis;

	assert(TermOne->numCols() == 1);
	TermOne->Transpose();
	TermTwo = (*TermOne) * X;
	TermOne->Transpose();
	gradient = TermTwo->Sum();
	assert (gradient->numCols() == params_to_derivate.numRows());

	gradient->operateOnMatrixValues(constant, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	gradient->Transpose();

	delete TermOne;
	delete TermTwo;
	delete &X;

	return gradient;
}

/* Regularization Operations */
float ML_SingleLogOps::computeCost(Matrix &parameters_to_evaluate)
{
	float unregularizedCost = this->computeCostInternal(parameters_to_evaluate);
	float regFactor = (float) ((float) lambda) / ((float) (2 * numTrainingExamples));
	Matrix &temp_theta = (*new Matrix(parameters_to_evaluate));

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

Matrix *ML_SingleLogOps::gradientCalculate(Matrix &params_to_derivate)
{
	Matrix &unregularizedGradients = (*this->gradientCalculateInternal(params_to_derivate));
	Matrix &temp_theta = (*new Matrix(params_to_derivate));
	float regFactor = (float) ((float) lambda) / ((float) (numTrainingExamples));

	/* Do not regularize the first parameter */
	temp_theta[0] = 0;


	temp_theta.operateOnMatrixValues(regFactor, OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT);

	Matrix *regularizedGradients = unregularizedGradients + temp_theta;

	delete &unregularizedGradients;
	delete &temp_theta;

	return regularizedGradients;
}

Matrix *ML_SingleLogOps::Optimize(Matrix &initial_params,  int num_iterations)
{
	/* myTheta = myTheta - ((alpha/m) * X' * (sigmoid(X * myTheta) - y)); */

	int iteration_idx = 0;
	Matrix *result = new Matrix(initial_params);
	Matrix *X = new Matrix(*this->data_x);
	X->AddBiasCol();
#if DEBUGGING
	float prevCost = 0;
#endif

	for (iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++)
	{
		Matrix *nextGradient = this->gradientCalculate(*result);
#if DEBUGGING
		/* When debugging, verify that gradient descent is actually decreasing/not increasing the cost */
		float currentCost = this->computeCost(training_X, training_y, theta, regularizationParam);
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

Matrix *ML_SingleLogOps::Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold)
{
	Matrix &input_examples = (*new Matrix(input_to_evaluate));
	input_examples.AddBiasCol();
	Matrix *interim_hypothesis = input_examples * parameters;
	Matrix *predictions = sigmoid(*(interim_hypothesis));
	predictions->operateOnMatrixValues(threshold, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GEQ_SCALAR);

	delete interim_hypothesis;
	return predictions;
}

/* Class IDs are in order */
Matrix *ML_SingleLogOps::OneVsAll(int num_iterations)
{
	int class_idx = 0;
	int feature_idx = 0;
	Matrix **all_theta = new Matrix*[this->numCategories];

	/*
	 * Return matrix has the same number of cols as number of classes to represent the optimal theta for each class to fit
	 * the given data, and the same number of columns as the training set + 1 since we have the same number of features as
	 * the original data set plus 1 for the bias feature.
	 */
	Matrix &return_matrix = (*new Matrix(this->numTrainingFeatures + 1, this->numCategories));

	for (class_idx = 0; class_idx < this->numCategories; class_idx++)
	{
		Matrix *temp_y = new Matrix(*this->data_y);
		Matrix *currentTheta = new Matrix(this->numTrainingFeatures + 1, 1);
		temp_y->operateOnMatrixValues(class_idx, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
		Matrix *temp_matrix = this->data_y;
		this->data_y = temp_y;
		Matrix *theta_idx = this->Optimize(*currentTheta, num_iterations);
		this->data_y = temp_matrix;
		all_theta[class_idx] = theta_idx;
		delete temp_y;
		delete currentTheta;
	}

	for (class_idx = 0; class_idx < this->numCategories; class_idx++)
	{
		for (feature_idx = 0; feature_idx < (this->numTrainingFeatures + 1); feature_idx++)
		{
			Indexer *currentFeatureForCurrentTheta = new Indexer(feature_idx, class_idx);
			return_matrix[currentFeatureForCurrentTheta] = (*all_theta[class_idx])[feature_idx];
			delete currentFeatureForCurrentTheta;
		}
	}

	delete[] all_theta;

	return &return_matrix;
}

Matrix *ML_SingleLogOps::PredictOneVsAll(Matrix &all_parameters)
{
	Matrix &X = (*new Matrix(*this->data_x));
	X.AddBiasCol();
	Matrix *hypothesis = X * all_parameters;
	Matrix *predict = this->sigmoid(*hypothesis);
	predict->Transpose();
	Matrix *end = predict->MaxRowNumber();
	return end;
	
}
