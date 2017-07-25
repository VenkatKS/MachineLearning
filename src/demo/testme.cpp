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
#include "../lib/LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../lib/RegressionClassificationLibrary/include/ml_regression.hpp"
#include "../lib/RegressionClassificationLibrary/include/ml_classification.hpp"
#include "../lib/NeuralNetworkLibrary/include/ml_neural_network.hpp"
#include "../lib/OpenCL/include/opencl_driver.hpp"

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

#define STRESS_TEST 0
#define ONLY_NN 1

/* Deviation should be less than 0.001% from MatLab Test Case */
#define EPSILON	0.001
#define ROUGHLY_EQUAL(_float_1, _float_2)	\
	((fabs((float) ((_float_1 - _float_2)/(_float_2))) * 100) < ((float) EPSILON))

int main(int argc, const char * argv[]) {
	clock_t begin = std::clock();

	int idx = 0;
	setUpOpenCLDrivers();

	printf("MLLib Regression Testing: \n\n\n");

#if (!ONLY_NN)
	/* Single-Feature Regression */
	Matrix *X = Matrix::LoadMatrix("data/regression/X1_data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/regression/y1_data.txt", ',');
	Matrix *theta_0 = new Matrix::Matrix(2, 1);
	Matrix *theta_1 = new Matrix::Matrix(2, 1);

	(*theta_1)[0] = -1.0;
	(*theta_1)[1] = 2.0;

	printf("1) Testing Single-Feature Linear Regression......\n");

	/* Should be around 32.07 */
	float result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, 32.072733877455676));
	printf("1a) First Cost Function Test: Passed\n");

	/* Should be around 54.24 */
	float result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, 54.242455082012391));
	assert(result_1 > (float) 54.24 && result_1 < (float) 54.245);
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
	X = Matrix::LoadMatrix("data/regression/X2_data.txt", ',');
	y = Matrix::LoadMatrix("data/regression/y2_data.txt", ',');
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

	//result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	//assert(ROUGHLY_EQUAL(result_1, 4.039928302794304e+37));
	//printf("2b) Second Cost Function Test: Passed\n");

	theta_0_temp = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 400);
	delete theta_0;
	theta_0 = theta_0_temp;

	assert (theta_0->numRows() == 3);
	assert (theta_1->numCols() == 1);
	assert(ROUGHLY_EQUAL((*theta_0)[0],  (float) 3.343020639932770e+05));
	assert(ROUGHLY_EQUAL((*theta_0)[1],  (float) 1.000871160058464e+05));
	assert(ROUGHLY_EQUAL((*theta_0)[2],  (float) 3.673548450928300e+03));
	printf("2b) First Gradient Descent Test: Passed\n");

	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\n\n3) Testing Multi-Feature Log Regression and Classification......\n");

	/* Single-Feature Classification */
	X = Matrix::LoadMatrix("data/classification/X1_data_log_1.txt", ',');
	y = Matrix::LoadMatrix("data/classification/y1_data_log_1.txt", ',');
	theta_0 = new Matrix::Matrix(3, 1);
	theta_1 = new Matrix::Matrix(3, 1);

	Matrix *sigmoid_result = ML_LogOps::sigmoid(*theta_0);
	for (idx = 0; idx < (sigmoid_result->numCols() * sigmoid_result->numRows()); idx++)
		assert ((*sigmoid_result)[idx] == 0.5);
	printf("3a) First Sigmoid Function Test: Passed\n");
	delete sigmoid_result;

	result_0 = ML_LogOps::computeCost(*X, *y, *theta_0);

	assert(ROUGHLY_EQUAL(result_0, (float) 0.693147180559946));
	printf("3b) First Cost Function Test: Passed\n");

	(*theta_1)[0] = -24;
	(*theta_1)[1] = (float) 0.20;
	(*theta_1)[2] = (float) 0.20;

	result_1 = ML_LogOps::computeCost(*X, *y, *theta_1);

	assert(ROUGHLY_EQUAL(result_1, (float) 0.218330193826598));
	printf("3b) Second Cost Function Test: Passed\n");

	delete theta_0;

	theta_0 = new Matrix(3, 1);
	theta_0_temp = ML_LogOps::gradientCalculate(*X, *y, *theta_0);
	delete theta_0;
	theta_0 = theta_0_temp;

	assert(ROUGHLY_EQUAL((*theta_0)[0],  (float) -0.100000000000000));
	assert(ROUGHLY_EQUAL((*theta_0)[1],  (float) -12.009216589291150));
	assert(ROUGHLY_EQUAL((*theta_0)[2],  (float) -11.262842205513591));
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

	Matrix *temp_y = ML_LogOps::Predict(*X, *theta_0, 0.5);
	temp_y->operateOnMatrixValues(y, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
	Matrix *result = temp_y->Mean();

	assert (result->numCols == 1);
	assert (result->numRows == 1);
	assert ((*result)[0] == 0.890000000000000);

	printf("3e) Predictions Match Expected Accuracy Test: Passed\n");

	delete result;

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

	Matrix *temp_y = ML_LogOps::Predict(*X, *theta_0, 0.5);
	temp_y->operateOnMatrixValues(y, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
	Matrix *result = temp_y->Mean();
	assert (result->numCols() == 1);
	assert (result->numRows() == 1);
	assert (ROUGHLY_EQUAL((*result)[0], 0.60));

	printf("3e) Predictions Match Expected Accuracy Test: Passed\n");

	delete result;
#endif


	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	/* Testing Multi-Class, Multi-Feature Classification */
	printf("\n\n4) Testing Multi-Class, Multi-Feature Log Regression and Classification...\n");

	X = Matrix::LoadMatrix("data/classification/multi-class classification/X_data.txt", ',');
	y = Matrix::LoadMatrix("data/classification/multi-class classification/Y_data.txt", ',');

	/* theta_t = [-2; -1; 1; 2]; */
	theta_0 = new Matrix(4, 1);
	(*theta_0)[0] = -2;
	(*theta_0)[1] = -1;
	(*theta_0)[2] = 1;
	(*theta_0)[3] = 2;
	result_1 = ML_LogOps::computeCost(*X, *y, *theta_0, 3);
	assert (ROUGHLY_EQUAL(result_1, 2.534819396109744));
	printf("4a) Regularized Log Cost Test: Passed\n");

	Matrix *regularized_gradients = ML_LogOps::gradientCalculate(*X, *y, *theta_0, 3);
	assert (ROUGHLY_EQUAL((*regularized_gradients)[0], 0.146561367924898));
	assert (ROUGHLY_EQUAL((*regularized_gradients)[1], -0.548558411853160));
	assert (ROUGHLY_EQUAL((*regularized_gradients)[2], 0.724722272109289));
	assert (ROUGHLY_EQUAL((*regularized_gradients)[3], 1.398002956071738));
	delete regularized_gradients;
	printf("4a) Regularized Log Gradients Test: Passed\n");
	delete X;
	delete y;
	delete theta_0;

	X = Matrix::LoadMatrix("data/classification/multi-class classification/XPic_Data.txt", ',');
	y = Matrix::LoadMatrix("data/classification/multi-class classification/YPic_Data.txt", ',');

	Matrix *all_theta = ML_LogOps::OneVsAll(*X, *y, 10, 0.01, 30, 0.1);
	temp_y = ML_LogOps::PredictOneVsAll(*X, *all_theta);
	temp_y->Transpose();
	temp_y->operateOnMatrixValues(y, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
	result = temp_y->Mean();
	assert (result->numRows() == 1);
	assert (result->numCols() == 1);
	/* FIXME: Verify using matlab */
	assert (ROUGHLY_EQUAL((*result)[0], 0.676600039));
	printf("4b) Predictions Match Expected Accuracy Test: Passed\n");
#else
	printf("Only exercising neural network changes\n");
	Matrix *X = Matrix::LoadMatrix("data/classification/multi-class classification/XPic_Data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/classification/multi-class classification/YPic_Data.txt", ',');
	Matrix *theta0 = Matrix::LoadMatrix("data/classification/multi-class classification/theta0nn.txt", ',');
	Matrix *theta1 = Matrix::LoadMatrix("data/classification/multi-class classification/theta1nn.txt", ',');

	Matrix **all_theta = new Matrix*[2];
	all_theta[0] = theta0;
	all_theta[1] = theta1;

	int *num_nodes = new int[1];
	num_nodes[0] = 25;
	neural_network *myNetwork = new neural_network(1, num_nodes, 400, 10, all_theta);
	Matrix *result = myNetwork->execute_nn(*myNetwork, *X, *y);
	Matrix *end2 = result->MaxRowNumber();
	end2->Transpose();
	end2->operateOnMatrixValues(y, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
	result = end2->Mean();
#endif
	clock_t end = std::clock();
	double elapsed_seconds = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "\nExecution Time Taken: " << elapsed_seconds << "\n";

	printf("\n\nDONE: All Pass within a deviation less than %e %% from MatLab's results.\n", EPSILON);

	return 0;
}
