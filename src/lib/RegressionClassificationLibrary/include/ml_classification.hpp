//
//  ml_classification.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 6/28/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_log_hpp
#define ml_log_hpp

#include <stdio.h>

/* Linear Regression Machine Learning Operations */
class ML_SingleLogOps:public MachineLearningFitModel
{
private:
	/* Unregularized operations used by public variants */
	float computeCostInternal(Matrix &parametes_to_evaluate);
	Matrix *gradientCalculateInternal(Matrix &params_to_derivate);
	int numCategories;

public:
	ML_SingleLogOps(DataSetWrapper *data, float learning_rate, float reg_rate, int numCategories) :
	MachineLearningFitModel(data, learning_rate, reg_rate)
	{
		this->numCategories = numCategories;
	}

	float computeCost(Matrix &parameters_to_evaluate);
	Matrix *gradientCalculate(Matrix &params_to_derivate);
	Matrix *Optimize(Matrix &initial_params,  int num_iterations);
	Matrix *sigmoid(Matrix &z);
	Matrix *Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold);

	/* Multi-Class Classification */
	Matrix *OneVsAll(int num_iterations);
	Matrix *PredictOneVsAll(Matrix &all_parameters);

	inline fit_category GetCategoryOfFit()
	{
		return LOG_SINGLE_CLASSIFICATION_MODEL;
	}
};

#endif /* ml_log_hpp */
