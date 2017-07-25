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
#include "../../LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../../MachineLearningLibrary.hpp"

/* Linear Regression Machine Learning Operations */
class ML_SingleLogOps:public MachineLearningFitModel
{
private:
	/* Unregularized operations used by public variants */
	float computeCostInternal(Matrix &parametes_to_evaluate);
	Matrix *gradientCalculateInternal(Matrix &params_to_derivate);

public:
	ML_SingleLogOps(DataSetWrapper *data, float learning_rate, float reg_rate) :
	MachineLearningFitModel(data, learning_rate, reg_rate)
	{
		/* Just call super */
	}

	float computeCost(Matrix &parameters_to_evaluate);
	Matrix *gradientCalculate(Matrix &params_to_derivate);
	Matrix *GradientDescent(Matrix &initial_params,  int num_iterations);
	Matrix *sigmoid(Matrix &z);
	Matrix *Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold);
};

class ML_MultiLogOps:public ML_SingleLogOps
{
protected:
	int numCategories;
public:
	ML_MultiLogOps(DataSetWrapper *data, float learning_rate, float reg_rate, int numCategories) :
	ML_SingleLogOps(data, learning_rate, reg_rate)
	{
		this->numCategories = numCategories;
	}
	/* Multi-Class Classification */
	Matrix *OneVsAll(int num_iterations);
	Matrix *PredictOneVsAll(Matrix &training_X, Matrix &all_parameters);


};

#endif /* ml_log_hpp */
