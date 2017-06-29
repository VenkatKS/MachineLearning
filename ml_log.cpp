//
//  ml_log.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/28/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <math.h>
#include "ml_log.hpp"

/* Performs the sigmoid function on each item in the provided matrix */
Matrix *ML_LogOps::sigmoid(Matrix &z)
{
	Matrix &Sigmoid_Matrix = *(new Matrix(z));
	Sigmoid_Matrix.MultiplyScalar(-1.0);

	double exponent = exp((double) 1.0);
	Sigmoid_Matrix.MatrixPower(exponent);
	Sigmoid_Matrix.AddScalar(1);
	Sigmoid_Matrix.ReciprocalMultiply(1.0);
	
	return &Sigmoid_Matrix;
}

double ML_LogOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
{

	return 0.0;
}
