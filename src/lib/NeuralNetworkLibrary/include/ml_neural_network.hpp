//
//  ml_neural_network.hpp
//  MachineLearning
//
//  Created by Venkat Srinivasan on 7/6/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#ifndef ml_neural_network_hpp
#define ml_neural_network_hpp

#include <stdio.h>
#include <cassert>

class neural_network
{
	/* Number of hidden layers */
	int hidden_layers;
	/* Number of nodes in each hidden layer */
	int *num_nodes;
	/* Number of inputs */
	int num_input;
	/* Number of outputs/classes */
	int num_classes;
	/* Parameters for each layer */
	Matrix **theta;
public:
	/* ASSUMPTION: Neural Network Is Strongly Connected In Each Layer. */
	neural_network(int numberOfHiddenLayers, int *numberOfNodesInEachLayer, int numberOfInputFeatures, int numberOfClasses,
		       Matrix **neuralNetworkParameters);
	Matrix *execute_nn(neural_network &network, Matrix &training_X, Matrix &training_y);
};

#endif /* ml_neural_network_hpp */
