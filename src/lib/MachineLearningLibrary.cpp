//
//  MachineLearningLibrary.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/7/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "MachineLearningLibrary.hpp"
#include "RegressionClassificationLibrary/include/ml_regression.hpp"
#include "RegressionClassificationLibrary/include/ml_classification.hpp"

/*
 * Constructor for a linear fit regression data model
 */
MachineLearning::MachineLearning(LinearRegressionFit &model)
{
	data_model = new ML_LinearOps(model.data, model.learning_rate, 0.01);
}

/*
 * Constructor for a log fit classification data model
 */
MachineLearning::MachineLearning(LogisiticClassificationFit &model)
{
	data_model = new ML_SingleLogOps(model.data, model.learning_rate, model.regularization_rate, model.numCategories);
}

/*
 * Constructor for a neural network data model
 */
MachineLearning::MachineLearning(NeuralNetworkFit &model)
{
	/* FIXME: Implement neural networks */
}

/*
 *	Assumptions:
 *		Matrix data is a dataset where the columns contain parameters/features and
 *		the rows contain the various training examples.
 */
void MachineLearningFitModel::NormalizeFeatureData()
{
	Matrix &data = *this->data_x;

	Matrix &Data_Normalized = *(new Matrix(data));
	int r_idx = 0;
	int c_idx = 0;

	/* Calculate the mean of each column (i.e. each feature) */
	Matrix &Data_Mean = (*data.Mean());
	/* Calculate the std. dev. of each column (i.e. each feature) */
	Matrix &Data_STD = *(data.StdDev());

	/* Normalize the array */
	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		Indexer *colMeanIndex = new Indexer(0, c_idx);
		float colMean = Data_Mean[colMeanIndex];
		float colStd = Data_STD[colMeanIndex];

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);

			float indexValue = data[currentIndex];
			indexValue = indexValue - colMean;
			indexValue = indexValue / colStd;

			Data_Normalized[currentIndex] = indexValue;
			delete currentIndex;
		}
		delete colMeanIndex;
	}

	delete &Data_Mean;
	delete &Data_STD;

	delete this->data_x;

	this->data_x = &Data_Normalized;
}


