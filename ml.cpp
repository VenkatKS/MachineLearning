//
//  ml.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "ml.hpp"


double ML_LinearOps::computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta)
{
	int numTrainingExamples = training_y.numRows();
	double currentCost = 0;
	int idx = 0;

	training_X.AddBiasCol();

	/* Calculate the hypothesis */
	Matrix *result = training_X * training_theta;

	if (!result)
	{
		printf("ERROR: Matrix dimensions between parameters and training examples must agree.\n");
		return -1;
	}

	result = (*result) - training_y;

	if (!result)
	{
		printf("ERROR: Matrix dimensions for training set invalid.\n");
		return -1;
	}

	result->PowerScalar(2);

	double divisor = (1.0 / (2.0 * (double) numTrainingExamples));
	for (idx = 0; idx < numTrainingExamples; idx++)
	{
		currentCost += divisor * (*result)[idx];
	}

	return currentCost;
}
