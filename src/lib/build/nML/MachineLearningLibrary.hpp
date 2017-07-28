//
//  MachineLearningLibrary.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/7/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef MachineLearningLibrary_hpp
#define MachineLearningLibrary_hpp

#include <stdio.h>

/*
 *	GENERAL ASSUMPTIONS:
 *		The library assumes a few things, and these assumptions apply to every function.
 *			1. All parameters/variables will be in seperate columns. For example, column 0 will represent X0,
 *			   column 1 will represent X1, etc.
 *				1a. The library itself will add the necessary bias columns as needed for each algorithm.
 *				    Please do not call add them yourself.
 *			2. All examples will be in seperate rows. Each row details a single training example for the learning
 *			   algorithm. The quantities returned throught the learning algorithm will also have the predicted
 *			   results for each example in the corresponding row.
 */

class MachineLearning;
class MachineLearningFitModel;

class DataSetWrapper
{
	/* Allows us to quickly access without needing to go through pesky and slow OOP getters/setters */
	friend MachineLearningFitModel;

private:
	Matrix *data_x = NULL;
	Matrix *data_y = NULL;

public:
	DataSetWrapper(Matrix *training_examples, Matrix *solutions)
	{
		this->data_x = (new Matrix(*training_examples));
		this->data_y = (new Matrix(*solutions));
	}

	DataSetWrapper(DataSetWrapper &other)
	{
		this->data_x = (new Matrix(*other.data_x));
		this->data_y = (new Matrix(*other.data_y));
	}

	~DataSetWrapper()
	{
		delete data_x;
		delete data_y;
	}

public:
	Matrix *getTrainingExamples()
	{
		/* protect our data from mods! */
		return (new Matrix(*data_x));
	}
	Matrix *getTrainingSolutions()
	{
		return (new Matrix(*data_y));
	}
};

typedef enum {
	LINEAR_REGRESSION_MODEL,
	LOG_SINGLE_CLASSIFICATION_MODEL,
	LOG_MULTI_CLASSIFICATION_MODEL,
	NEURAL_NETWORK_MODEL
} fit_category;

class MachineLearningFitModel
{
protected:
	/* Training examples and solutions */
	DataSetWrapper *examples = NULL;
	int numTrainingExamples;
	int numTrainingFeatures;
	/* Allows for inherited classes to easily access (friendship isn't inherited) */
	Matrix *data_x = NULL;
	Matrix *data_y = NULL;

	/* Learning Rate */
	double alpha;
	/* Regularization Rate */
	double lambda;

	/*
	 * The training examples provided here will be used for every learning algorithm you call using this object.
	 * Only the static "Predict" function will require external training/feature data. If you do not wish to use
	 * regularization, please provide 0.0 for the reg_rate value. The ideal learning rate is 0.01. The ideal
	 * regularization rate is 0.1;
	 */
	MachineLearningFitModel(DataSetWrapper *data, float learning_rate, float reg_rate)
	{
		/* Initialize the member units properly */
		this->examples = new DataSetWrapper(*data);
		this->data_x = this->examples->data_x;
		this->numTrainingExamples = this->examples->data_x->numRows();
		this->numTrainingFeatures = this->examples->data_x->numCols();
		this->data_y = this->examples->data_y;
		this->alpha = learning_rate;
		this->lambda =  reg_rate;
	}

public:
	/*
	 * Computes the cost of the current parameters for the provided data set.
	 */
	virtual float computeCost(Matrix &parameters_to_evaluate) = 0;
	/*
	 * Performs the partial derivative of the respective cost function in respect for each provided variable.
	 * The gradient descent algorithm calls into this to calculate the partial derivatives during each iteration.
	 */
	virtual Matrix *gradientCalculate(Matrix &params_to_derivate) = 0;
	/*
	 * Performs the Gradient Descent algorithm on your data in order to calculate the
	 * best fit parameters for your provided data set (within the specified iteration bounds).
	 * The higher the max number of iterations, the slower the processing time, but the better the fit.
	 * This is a very computationally intensive algorithm, so much of it is parallelized using OPENCL.
	 */
	virtual Matrix *Optimize(Matrix &initial_params,  int num_iterations) = 0;
	/*
	 * Given the provided parameters, returns the predicted results for each example.
	 */
	virtual Matrix *Predict(Matrix &input_to_evaluate, Matrix &parameters, float threshold) = 0;

	inline virtual fit_category GetCategoryOfFit() = 0;

	/* Normalize the Matrix to allow for more even weighted features */
	void NormalizeFeatureData();
};

typedef struct LinearRegressionFit {
	DataSetWrapper *data = NULL;
	/* Ideal is 0.01 */
	float learning_rate = 0.01;
	LinearRegressionFit(DataSetWrapper* data, float learning_rate) : data(data), learning_rate(learning_rate)
	{
	}
} LinearRegressionFit;

typedef struct LogisiticClassificationFit {
	DataSetWrapper *data = NULL;
	/* Number of classification categories */
	int numCategories;
	/* Ideal is 0.01 */
	float learning_rate = 0.01;
	/* Ideal is 0.01 */
	float regularization_rate = 0.1;

	LogisiticClassificationFit(DataSetWrapper* data, int numCategories, float learning_rate, float regularization_rate) : data(data), numCategories(numCategories), learning_rate(learning_rate), regularization_rate(regularization_rate)
	{
	}
} LogisiticClassificationFit;

typedef struct {

} NeuralNetworkFit;


/*
 * Use this object to fit your data if you are wanting to use a linear regression fit for continuous values.
 */
class MachineLearning
{
protected:
	MachineLearningFitModel *data_model;
public:
	MachineLearning(LinearRegressionFit &model);
	MachineLearning(LogisiticClassificationFit &model);
	MachineLearning(NeuralNetworkFit &model);

	inline MachineLearningFitModel *Algorithms()
	{
		return data_model;
	}
};
#endif /* MachineLearningLibrary_hpp */
