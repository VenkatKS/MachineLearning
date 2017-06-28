//
//  main.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/25/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <iostream>
#include "2DMatrix.hpp"
#include "ml.hpp"

int main(int argc, const char * argv[]) {
	Matrix *X = Matrix::LoadMatrix("X1_data.txt");
	Matrix *y = Matrix::LoadMatrix("y1_data.txt");
	Matrix *theta = new Matrix::Matrix(2, 1);

	double result = ML_LinearOps::computeCost(*X, *y, *theta);

	return 0;
}
