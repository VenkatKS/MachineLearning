//
//  main.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/25/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <iostream>
#include "2DMatrix.hpp"
#include "ml_linear.hpp"

int main(int argc, const char * argv[]) {
	Matrix *X = Matrix::LoadMatrix("X1_data.txt");
	Matrix *y = Matrix::LoadMatrix("y1_data.txt");
	Matrix *theta_0 = new Matrix::Matrix(2, 1);
	Matrix *theta_1 = new Matrix::Matrix(2, 1);

	(*theta_1)[0] = -1.0;
	(*theta_1)[1] = 2.0;

	ML_LinearOps nonRegLinear = new ML_LinearOps(false);

	/* Should be around 32.07 */
	double result_0 = nonRegLinear.computeCost(*X, *y, *theta_0);

	/* Should be around 54.24 */
	double result_1 = nonRegLinear.computeCost(*X, *y, *theta_1);

	/* Should be about X0 = -3.6303, X1 = 1.1664 */
	theta_0 = nonRegLinear.gradientDescent(*X, *y, *theta_0, 0.0100, 1500);

	delete X;
	delete y;
	delete theta_0;

	X = Matrix::LoadMatrix("X2_data.txt");
	y = Matrix::LoadMatrix("y2_data.txt");
	theta_0 = new Matrix::Matrix(3, 1);
	theta_1 = new Matrix::Matrix(3, 1);

	ML_DataOps::NormalizeData(*X);

	return 0;
}
