//
//  main.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/25/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <cassert>
#include <math.h>
#include "../lib/include/2DMatrix.hpp"
#include "../lib/include/ml_linear.hpp"
#include "../lib/include/ml_log.hpp"

/* Deviation should be less than 0.00000000001% from MatLab Test Case */
#define EPSILON	0.00000000001
#define ROUGHLY_EQUAL(_double_1, _double_2)	\
	((fabs((double) ((_double_1 - _double_2)/(_double_2))) * 100) < ((double) EPSILON))

int main(int argc, const char * argv[]) {

	int idx = 0;

	printf("MLLib Regression Testing: \n\n\n");

	/* Single-Feature Regression */
	Matrix *X = Matrix::LoadMatrix("data/regression/X1_data.txt");
	Matrix *y = Matrix::LoadMatrix("data/regression/y1_data.txt");
	Matrix *theta_0 = new Matrix::Matrix(2, 1);
	Matrix *theta_1 = new Matrix::Matrix(2, 1);

	(*theta_1)[0] = -1.0;
	(*theta_1)[1] = 2.0;

	printf("1) Testing Single-Feature Linear Regression......\n");

	/* Should be around 32.07 */
	double result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, 32.072733877455676));
	printf("1a) First Cost Function Test: Passed\n");

	/* Should be around 54.24 */
	double result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, 54.242455082012391));
	assert(result_1 > (double) 54.24 && result_1 < (double) 54.245);
	printf("1b) Second Cost Function Test: Passed\n");

	/* Should be about X0 = -3.6303, X1 = 1.1664 */
	theta_0 = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 1500);

	/* Ensure it's a 2x1 Matrix with proper values */
	assert (theta_0->numRows() == 2);
	assert (theta_0->numCols() == 1);
	assert(ROUGHLY_EQUAL((*theta_0)[0], -3.630291439404359));
	assert(ROUGHLY_EQUAL((*theta_0)[1],  1.166362350335582));

	printf("1c) First Gradient Descent Test: Passed\n");

	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\n\n2) Testing Multi-Feature Linear Regression......\n");

	/* Multi-Feature Regression */
	X = Matrix::LoadMatrix("data/regression/X2_data.txt");
	y = Matrix::LoadMatrix("data/regression/y2_data.txt");
	theta_0 = new Matrix::Matrix(3, 1);
	theta_1 = new Matrix::Matrix(3, 1);

	(*theta_1)[0] = -0.324235453453425;
	(*theta_1)[1] = 0.3242345345234562345634253423452345;
	(*theta_1)[2] = 9085981324123412341;

	X = ML_DataOps::NormalizeData(*X);

	result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, 6.559154810645744e+10));
	printf("2a) First Cost Function Test: Passed\n");


	result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, 4.039928302794304e+37));
	printf("2b) Second Cost Function Test: Passed\n");


	theta_0 = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 400);

	assert (theta_0->numRows() == 3);
	assert (theta_1->numCols() == 1);
	assert(ROUGHLY_EQUAL((*theta_0)[0],  (double) 3.343020639932770e+05));
	assert(ROUGHLY_EQUAL((*theta_0)[1],  (double) 1.000871160058464e+05));
	assert(ROUGHLY_EQUAL((*theta_0)[2],  (double) 3.673548450928300e+03));
	printf("2c) First Gradient Descent Test: Passed\n");

	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\n\n3) Testing Multi-Feature Log Regression and Classification......\n");

	/* Single-Feature Classification */
	X = Matrix::LoadMatrix("data/classification/X1_data_log_1.txt");
	y = Matrix::LoadMatrix("data/classification/y1_data_log_1.txt");
	theta_0 = new Matrix::Matrix(3, 1);
	theta_1 = new Matrix::Matrix(3, 1);

	Matrix *sigmoid_result = ML_LogOps::sigmoid(*theta_0);
	for (idx = 0; idx < (sigmoid_result->numCols() * sigmoid_result->numRows()); idx++)
		assert ((*sigmoid_result)[idx] == 0.5);
	printf("3a) First Sigmoid Function Test: Passed\n");

	result_0 = ML_LogOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, (double) 0.693147180559946));
	printf("3b) First Cost Function Test: Passed\n");

	printf("\n\nDONE: All Pass within a deviation of %e percent from MatLab's results.\n", EPSILON);
	return 0;
}
