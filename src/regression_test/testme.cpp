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
#include "lib/include/2DMatrix.hpp"
#include "lib/include/ml_linear.hpp"
#include "lib/include/ml_log.hpp"

/*
 *	PLEASE NOTE:
 *		PLEASE NOTE THAT THIS IS NOT A PART OF THE MACHINE LEARNING LIBRARY.
 *		THIS IS A REGRESSION TESTING SUITE THAT VERIFIES ANY CHANGES MADE TO
 *		THE LIBRARY CODE. IT'S DESIGNED TO ENSURE THAT IT'S WITHIN A VERY SMALL
 *		TOLERANCE OF WHAT THE EQUIVALENT MATLAB RESULTS WOULD BE (WITHIN 0.00000001%).
 *		YOU CAN SAFELY DISREGARD THIS FILE IF YOU ARE NOT INTERESTED IN MAKING CHANGES
 *		TO THE LIBRARY CODE.
 */



/* Uncomment this to run stress tests */
/*
#define STRESS_TEST
*/

/* Deviation should be less than 0.00000001% from MatLab Test Case */
#define EPSILON	0.00000001
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
	Matrix *theta_0_temp;
	theta_0_temp = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 1500);
	delete theta_0;
	theta_0 = theta_0_temp;

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

	Matrix *temp_X;
	temp_X = ML_DataOps::NormalizeData(*X);
	delete X;
	X = temp_X;

	result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, 6.559154810645744e+10));
	printf("2a) First Cost Function Test: Passed\n");


	result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, 4.039928302794304e+37));
	printf("2b) Second Cost Function Test: Passed\n");


	theta_0_temp = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 400);
	delete theta_0;
	theta_0 = theta_0_temp;

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
	delete sigmoid_result;

	result_0 = ML_LogOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, (double) 0.693147180559946));
	printf("3b) First Cost Function Test: Passed\n");

	(*theta_1)[0] = -24;
	(*theta_1)[1] = (double) 0.20;
	(*theta_1)[2] = (double) 0.20;

	result_1 = ML_LogOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, (double) 0.218330193826598));
	printf("3b) Second Cost Function Test: Passed\n");

	delete theta_0;

	theta_0 = new Matrix(3, 1);
	theta_0_temp = ML_LogOps::gradientCalculate(*X, *y, *theta_0);
	delete theta_0;
	theta_0 = theta_0_temp;

	assert(ROUGHLY_EQUAL((*theta_0)[0],  (double) -0.100000000000000));
	assert(ROUGHLY_EQUAL((*theta_0)[1],  (double) -12.009216589291150));
	assert(ROUGHLY_EQUAL((*theta_0)[2],  (double) -11.262842205513591));
	printf("3c) Log Individual Gradient Test: Passed\n");

	delete theta_0;
	theta_0 = new Matrix(3, 1);

#if STRESS_TEST
	printf("Stress Test Enabled. This will take a while.....\n");
	theta_0_temp = ML_LogOps::GradientDescent(*X, *y, *theta_0, 0.001, 3000000);
	delete theta_0;
	theta_0 = theta_0_temp;
	result_1 = ML_LogOps::computeCost(*X, *y, *theta_0);

	assert (theta_0->numRows() == 3);
	assert (theta_0->numCols() == 1);
	assert (ROUGHLY_EQUAL((*theta_0)[0], -21.067462449096109));
	assert (ROUGHLY_EQUAL((*theta_0)[1], 0.173509794273152));
	assert (ROUGHLY_EQUAL((*theta_0)[2], 0.168334317694033));
	assert (ROUGHLY_EQUAL(result_1, 0.206392658299166));
	printf("3d) Stress Log Gradient Descent Test: Passed.\n");

#else
	theta_0_temp = ML_LogOps::GradientDescent(*X, *y, *theta_0, 0.001, 10000);
	delete theta_0;
	theta_0 = theta_0_temp;
	result_1 = ML_LogOps::computeCost(*X, *y, *theta_0);

	assert (theta_0->numRows() == 3);
	assert (theta_0->numCols() == 1);
	assert (ROUGHLY_EQUAL((*theta_0)[0], -0.668966029322792));
	assert (ROUGHLY_EQUAL((*theta_0)[1], 0.015091499161493));
	assert (ROUGHLY_EQUAL((*theta_0)[2], 0.005662370305905));
	assert (ROUGHLY_EQUAL(result_1, 0.585027498817674));
	printf("3d) Non-Stress Log Gradient Descent Test: Passed.\n");
#endif
	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\n\nDONE: All Pass within a deviation less than %e %% from MatLab's results.\n", EPSILON);
	return 0;
}
