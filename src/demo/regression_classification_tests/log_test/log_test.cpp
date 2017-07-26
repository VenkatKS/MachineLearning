//
//  log_test.cpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/17/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "log_test.hpp"
#include "../../../lib/LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../../../lib/MachineLearningLibrary.hpp"
#include "../../../lib/RegressionClassificationLibrary/include/ml_classification.hpp"

#include <cassert>

bool log_regression_test::run_single_feature_test()
{
	int idx = 0;

	float expected_results[] =
	{
		/* Expected output for Sigmoid function with all 0s as input */
		0.5,
		/* Cost for unregularized first test with all 0s as parameters */
		0.693147180559946,
		/* Cost for unregularized second test with specific input parameters */
		0.218330193826598,
		/* Unregularized single gradients */
		-0.100000000000000,
		-12.009216589291150,
		-11.262842205513591,
		/* Unregularized optimized parameters within 3MIL iterations, with all 0s initial params */
		-0.668966029322792,
		0.015091499161493,
		0.005662370305905,
		/* Unregularized cost for optimized parameters within 3 million iterations */
		0.585027498817674,
		/* Prediction accuracy */
		0.6
	};

	Matrix *X = Matrix::LoadMatrix("data/Classification/X1_data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/Classification/y1_data.txt", ',');
	float result = 0.0;

	Matrix *theta_0 = new Matrix::Matrix(3, 1);
	Matrix *theta_1 = new Matrix::Matrix(3, 1);

	/* Create our machine learning object using the loaded data as our operating data set */
	DataSetWrapper *test_wrapper = new DataSetWrapper(X, y);

	/* Create a logistic regression fit model */
	LogisiticClassificationFit *logfit = new LogisiticClassificationFit(test_wrapper, 10, 0.001, 0);
	MachineLearning *logisticOperations = new MachineLearning(*logfit);
	ML_SingleLogOps *algoModel =  (ML_SingleLogOps *)logisticOperations->Algorithms();

	/* Sigmoid function is unique to only the log classification model, so cast the object */
	Matrix *sigmoid_result = algoModel->sigmoid(*theta_0);
	for (idx = 0; idx < (sigmoid_result->numCols() * sigmoid_result->numRows()); idx++)
		if (!((*sigmoid_result)[idx] == expected_results[0])) goto clean_up_false;
	delete sigmoid_result;

	result = logisticOperations->Algorithms()->computeCost(*theta_0);

	if (!(this->roughly_equal(result, expected_results[1])))
		goto clean_up_false;

	(*theta_1)[0] = -24;
	(*theta_1)[1] = (float) 0.20;
	(*theta_1)[2] = (float) 0.20;

	result = logisticOperations->Algorithms()->computeCost(*theta_1);

	if (!(this->roughly_equal(result, (float) expected_results[2])))
		goto clean_up_false;

	delete theta_0;
	delete theta_1;

	/* FIXME: Memory leak */
	theta_0 = new Matrix(3, 1);
	theta_0 = logisticOperations->Algorithms()->gradientCalculate(*theta_0);

	if (!(this->roughly_equal((*theta_0)[0], (float) expected_results[3])))
		goto clean_up_false;

	if (!(this->roughly_equal((*theta_0)[1], (float) expected_results[4])))
		goto clean_up_false;

	if (!(this->roughly_equal((*theta_0)[2], (float) expected_results[5])))
		goto clean_up_false;

	delete theta_0;

	/* FIXME: Memory leak */
	theta_0 = new Matrix(3, 1);
	theta_0 = logisticOperations->Algorithms()->GradientDescent(*theta_0, 10000);
	result = logisticOperations->Algorithms()->computeCost(*theta_0);
	assert (theta_0->numRows() == 3);
	assert (theta_0->numCols() == 1);

	if (!(this->roughly_equal((*theta_0)[0], (float) expected_results[6])))
		goto clean_up_false;

	if (!(this->roughly_equal((*theta_0)[1], (float) expected_results[7])))
		goto clean_up_false;

	if (!(this->roughly_equal((*theta_0)[2], (float) expected_results[8])))
		goto clean_up_false;

	if (!(this->roughly_equal(result, (float) expected_results[9])))
		goto clean_up_false;

	/* FIXME: Memory leak */
	theta_0 = logisticOperations->Algorithms()->Predict(*X, *theta_0, 0.5);
	theta_0->operateOnMatrixValues(y, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
	theta_0 = theta_0->Mean();

	assert (theta_0->numCols() == 1);
	assert (theta_0->numRows() == 1);
	if (!(this->roughly_equal((*theta_0)[0], (float) expected_results[10])))
		goto clean_up_false;

	delete logisticOperations;

	return true;

clean_up_false:
	delete X;
	delete y;
	delete theta_0;
	return false;
}

bool log_regression_test::run_multi_feature_test()
{
	float expected_results[] =
	{
		/* Expected cost for dataset fitted with pre-selected parameters */
		2.534819396109744,

		/* Expected optimized parameters for dataset */
		0.146561367924898,
		-0.548558411853160,
		0.724722272109289,
		1.398002956071738
	};

	Matrix *X = Matrix::LoadMatrix("data/Classification/X2_data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/Classification/y2_data.txt", ',');
	Matrix *regularized_gradients = NULL;

	/* Create our machine learning object using the loaded data as our operating data set */
	DataSetWrapper *test_wrapper = new DataSetWrapper(X, y);

	/* Create a linear regression fit model */
	LogisiticClassificationFit *logfit = new LogisiticClassificationFit(test_wrapper, 10, 0.001, 3);
	MachineLearning *logisticOperations = new MachineLearning(*logfit);

	/* theta_t = [-2; -1; 1; 2]; */
	Matrix *theta_0 = new Matrix(4, 1);
	(*theta_0)[0] = -2;
	(*theta_0)[1] = -1;
	(*theta_0)[2] = 1;
	(*theta_0)[3] = 2;
	float result = logisticOperations->Algorithms()->computeCost(*theta_0);

	if (!(this->roughly_equal(result, (float) expected_results[0])))
		goto clean_up_false;

	regularized_gradients = logisticOperations->Algorithms()->gradientCalculate(*theta_0);

	if (!(this->roughly_equal((*regularized_gradients)[0], (float) expected_results[1])))
		goto clean_up_false;

	if (!(this->roughly_equal((*regularized_gradients)[1], (float) expected_results[2])))
		goto clean_up_false;

	if (!(this->roughly_equal((*regularized_gradients)[2], (float) expected_results[3])))
		goto clean_up_false;

	if (!(this->roughly_equal((*regularized_gradients)[3], (float) expected_results[4])))
		goto clean_up_false;

	delete regularized_gradients;

	return true;

clean_up_false:
	delete X;
	delete y;
	delete theta_0;
	return false;

}

bool log_regression_test::run_test()
{
	/* Run 2 variable linear test -- 1 feature, 1 bias */
	bool result = true;

	result = this->run_single_feature_test();

	if (result)
		printf("Single-Feature Log Regression Test Pass.\n");
	else
		assert (0);

	/* Run 3 variable linear test -- 2 features, 1 bias */
	result = this->run_multi_feature_test();

	if (result)
		printf("Multi-Feature Log Regression Test Pass.\n");
	else
		assert (0);

	return result;
}
