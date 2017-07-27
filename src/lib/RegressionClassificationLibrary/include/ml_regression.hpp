//
//  ml.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/26/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_hpp
#define ml_hpp

#include <stdio.h>

/* Linear Regression Machine Learning Operations */
class ML_LinearOps: public MachineLearningFitModel
{
public:
	ML_LinearOps(DataSetWrapper *data, float learning_rate, float reg_rate) :
	MachineLearningFitModel(data, learning_rate, reg_rate)
	{
		/* Just call super */
	}

	/* Explanations for these functions can be found in the master file, MachineLearningLibrary.hpp */
	float computeCost(Matrix &parameters_to_evaluate);
	Matrix *gradientCalculate(Matrix &params_to_derivate);
	Matrix *Optimize(Matrix &initial_params,  int num_iterations);
	Matrix *Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold);

	inline fit_category GetCategoryOfFit()
	{
		return LINEAR_REGRESSION_MODEL;
	}
};
#endif /* ml_hpp */
