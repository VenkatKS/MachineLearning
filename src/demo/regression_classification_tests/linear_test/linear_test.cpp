//
//  linear_regression_test.cpp
//  MachineLearningLibrary
//
//  Created by Venkat Srinivasan on 7/11/17.
//  Copyright © 2017 Venkat Srinivasan. All rights reserved.
//

#include <cassert>
#include "linear_test.hpp"
#include "../../../lib/LinearAlgebraLibrary/include/2DMatrix.hpp"
#include "../../../lib/MachineLearningLibrary.hpp"


bool linear_regression_test::run_single_feature_test()
{
	float expected_results[] =
	{
		/* Expected cost computed for dataset using 2 features, all initialized to 0 */
		32.072733877455676,
		/* Expected cost computed for dataset using 2 features, initialized to {-1.0, 2.0} */
		54.242455082012391,
		/* Expected optimized value for feature 0 after 1500 iterations of gradient descent using all 0s init */
		-3.630291439404359,
		/* Expected optimized value for feature 1 after 1500 iterations of gradient descent using all 0s init */
		1.166362350335582
	};

	bool match = true;

	/* Load the data from the files into Matrix objects, using comma as the seperator for cols and newlines for rows */
	Matrix *X = Matrix::LoadMatrix("data/Regression/X1_data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/Regression/y1_data.txt", ',');
	Matrix *theta_result = NULL;

	/* Create our feature vectors. Our files have 2 features/cols, so we make a 2 element col vector */
	Matrix *theta_0 = new Matrix::Matrix(2, 1);

	/* Create our machine learning object using the loaded data as our operating data set */
	DataSetWrapper *test_wrapper = new DataSetWrapper(X, y);

	/* Create a linear regression fit model */
	LinearRegressionFit *linfit = new LinearRegressionFit(test_wrapper, 0.01);
	MachineLearning *linearOperations = new MachineLearning(*linfit);

	float result = linearOperations->Algorithms()->computeCost(*theta_0);

	if (!(this->roughly_equal(result, expected_results[0])))
		goto clean_up_false;

	/* For testing, initialize parameters to other values */
	(*theta_0)[0] = -1.0;
	(*theta_0)[1] = 2.0;

	result = linearOperations->Algorithms()->computeCost(*theta_0);

	if (!(this->roughly_equal(result, expected_results[1])))
		goto clean_up_false;

	(*theta_0)[0] = 0.0;
	(*theta_0)[1] = 0.0;
	theta_result = linearOperations->Algorithms()->GradientDescent(*theta_0, 1500);

	match &= (theta_result->numRows() == 2);
	match &= (theta_result->numCols() == 1);
	match &= (this->roughly_equal((*theta_result)[0], expected_results[2]));
	match &= (this->roughly_equal((*theta_result)[1], expected_results[3]));

	if (!match)
		goto clean_up_false;

	delete theta_result;


	delete X;
	delete y;
	delete theta_0;
	return true;

clean_up_false:
	delete X;
	delete y;
	delete theta_0;
	return false;
}

bool linear_regression_test::run_multi_feature_test()
{
	float expected_results[] =
	{
		/* Expected cost for multi-feature linear regression with features set to all 0s */
		6.559154810645744e+10,
		/* Optimal feature values for gradient descent, starting at all features set at 0 for 400 iterations */
		3.343020639932770e+05,
		1.000871160058464e+05,
		3.673548450928300e+03
	};

	Matrix *X = Matrix::LoadMatrix("data/Regression/X2_data.txt", ',');
	Matrix *y = Matrix::LoadMatrix("data/Regression/y2_data.txt", ',');
	float result = 0.0;
	bool match = true;

	Matrix *theta_0 = new Matrix::Matrix(3, 1);
	Matrix *theta_1 = NULL;

	/* Create our machine learning object using the loaded data as our operating data set */
	DataSetWrapper *test_wrapper = new DataSetWrapper(X, y);

	/* Create a linear regression fit model */
	LinearRegressionFit *linfit = new LinearRegressionFit(test_wrapper, 0.01);
	MachineLearning *linearOperations = new MachineLearning(*linfit);

	linearOperations->Algorithms()->NormalizeFeatureData();

	result = linearOperations->Algorithms()->computeCost(*theta_0);

	if (!(this->roughly_equal(result, expected_results[0])))
		goto clean_up_false;

	theta_1 = linearOperations->Algorithms()->GradientDescent(*theta_0, 400);

	match = match & (theta_1->numRows() == 3);
	match = match & (theta_1->numCols() == 1);
	match = match & this->roughly_equal((*theta_1)[0], expected_results[1]);
	match = match & this->roughly_equal((*theta_1)[1], expected_results[2]);
	match = match & this->roughly_equal((*theta_1)[2], expected_results[3]);

	if (!match)
		goto clean_up_false;

	return true;

clean_up_false:
	delete X;
	delete y;
	delete theta_0;
	return false;


}

bool linear_regression_test::run_test()
{
	/* Run 2 variable linear test -- 1 feature, 1 bias */
	bool result = true;

	result = this->run_single_feature_test();

	if (result)
		printf("Single-Feature Linear Regression Test Pass.\n");
	else
		assert (0);

	/* Run 3 variable linear test -- 2 features, 1 bias */
	result = this->run_multi_feature_test();

	if (result)
		printf("Multi-Feature Linear Regression Test Pass.\n");
	else
		assert (0);

	return result;
}
