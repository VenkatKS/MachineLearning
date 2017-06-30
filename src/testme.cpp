//
//  main.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/25/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include "lib/include/2DMatrix.hpp"
#include "lib/include/ml_linear.hpp"
#include "lib/include/ml_log.hpp"

int main(int argc, const char * argv[]) {

	printf("MLLib Regression Testing.\n");

	/* Single-Feature Regression */
	Matrix *X = Matrix::LoadMatrix("data/regression/X1_data.txt");
	Matrix *y = Matrix::LoadMatrix("data/regression/y1_data.txt");
	Matrix *theta_0 = new Matrix::Matrix(2, 1);
	Matrix *theta_1 = new Matrix::Matrix(2, 1);

	(*theta_1)[0] = -1.0;
	(*theta_1)[1] = 2.0;

	printf("Testing Single-Feature Linear Regression.\n");

	/* Should be around 32.07 */
	double result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	/* Should be around 54.24 */
	double result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	printf("Cost: All Zeroes: %f, [-1.0, 2.0]: %f\n", result_0, result_1);
	printf("Expected: All Zeroes: 32.07, [-1.0, 2.0]: 54.24\n");

	/* Should be about X0 = -3.6303, X1 = 1.1664 */
	theta_0 = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 1500);

	Matrix::printMatrix(theta_0);
	printf("\nExpected: [-3.6303, 1.1664]\n");

	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\nTesting Multi-Feature Linear Regression\n");

	/* Multi-Feature Regression */
	X = Matrix::LoadMatrix("data/regression/X2_data.txt");
	y = Matrix::LoadMatrix("data/regression/y2_data.txt");
	theta_0 = new Matrix::Matrix(3, 1);
	theta_1 = new Matrix::Matrix(3, 1);

	X = ML_DataOps::NormalizeData(*X);

	result_0 = ML_LinearOps::computeCost(*X, *y, *theta_0);

	result_1 = ML_LinearOps::computeCost(*X, *y, *theta_1);

	theta_0 = ML_LinearOps::gradientDescent(*X, *y, *theta_0, 0.0100, 400);

	delete X;
	delete y;
	delete theta_0;
	delete theta_1;

	printf("\nTesting Multi-Feature Log Regression and Classification\n");

	/* Multi-Feature Classification */
	theta_0 = new Matrix(100, 1);
	Matrix *sigmoid_result = ML_LogOps::sigmoid(*theta_0);
	
	return 0;
}
