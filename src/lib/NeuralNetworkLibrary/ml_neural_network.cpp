//
//  ml_neural_network.cpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/6/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include "../LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../MachineLearningLibrary.hpp"
#include "../RegressionClassificationLibrary/include/ml_classification.hpp"
#include "include/ml_neural_network.hpp"




neural_network::neural_network(int numberOfHiddenLayers, int *numberOfNodesInEachLayer, int numberOfInputFeatures, int numberOfClasses,
	       Matrix **neuralNetworkParameters)
{
	int idx = 0;

	this->hidden_layers = numberOfHiddenLayers;
	this->num_nodes = numberOfNodesInEachLayer;
	this->num_input = numberOfInputFeatures;
	this->num_classes = numberOfClasses;
	this->theta = neuralNetworkParameters;

	/* Verify that the proper information is set for the provided parameters */
	assert (theta[0]->numCols() == (numberOfInputFeatures + 1));
	assert (theta[0]->numRows() == numberOfNodesInEachLayer[0]);
	for (idx = 1; idx < numberOfHiddenLayers; idx++)
	{
		assert (theta[idx]->numCols() == numberOfNodesInEachLayer[idx]);
		assert (theta[idx]->numRows() == numberOfNodesInEachLayer[idx + 1]);
	}
	assert (theta[numberOfHiddenLayers]->numRows() == numberOfClasses);
	assert (theta[numberOfHiddenLayers]->numCols() == (numberOfNodesInEachLayer[numberOfHiddenLayers - 1] + 1));


	this->theta = neuralNetworkParameters;

}


/* FIXME: Utilize OPENCL to accelerate NN processing */
Matrix *neural_network::execute_nn(neural_network &network, Matrix &training_X, Matrix &training_y)
{
//	int hiddenLayerIdx = 0;
//	Matrix *z = NULL;
//
//	z = new Matrix(training_X);
//
//	z->AddBiasCol();
//	z->Transpose();
//	Matrix *hypothesis = (*this->theta[0]) * (*z);
//	delete z;
//	z = ML_LogOps::sigmoid(*hypothesis);
//	z->Transpose();
//
//	z->AddBiasCol();
//	z->Transpose();
//	hypothesis = (*this->theta[1]) * (*z);
//	z = ML_LogOps::sigmoid(*hypothesis);

	return NULL;
}
