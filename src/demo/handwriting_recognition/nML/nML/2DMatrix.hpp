//
//  Matrix.hpp
//
//
//  Created by Venkat Srinivasan on 6/25/17.
//
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

/* Used to index into the array */
class Indexer
{
public:
	int rowID = 0;
	int colID = 0;

	Indexer(int rowID, int colID)
	{
		this->rowID = rowID;
		this->colID = colID;
	}
};

enum ScalarOps
{
	/* Scalar Operations To Do To Each Element */

	/* Matrix[i, j] = Matrix[i, j] + scalar; */
	OP_ADD_SCALAR_TO_EVERY_MATRIX_ELEMENT,

	/* Matrix[i, j] = Matrix[i, j] - scalar; */
	OP_SUBTRACT_SCALAR_FROM_EVERY_MATRIX_ELEMENT,

	/* Matrix[i, j] = Matrix[i, j] * scalar; */
	OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT,

	/* Matrix[i, j] = Matrix[i, j] ^ scalar; */
	OP_RAISE_EVERY_MATRIX_ELEMENT_TO_SCALAR_POWER,

	/* Matrix[i, j] = scalar - Matrix[i, j]; */
	OP_SUBTRACT_EVERY_MATRIX_ELEMENT_FROM_SCALAR,

	/* Matrix[i, j] = scalar ^ Matrix[i, j]; */
	OP_RAISE_SCALAR_TO_EVERY_MATRIX_ELEMENT_POWER,

	/* MatrixVal = (1/MatrixVal) * scalar; */
	OP_INVERT_EVERY_MATRIX_ELEMENT_AND_MULTIPLY_SCALAR
};

enum BooleanOps
{
	/* Matrix[i, j] = Matrix[i, j] == scalar; */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR,

	/* Matrix[i, j] = Matrix[i, j] != scalar */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_NOT_EQUAL_TO_SCALAR,

	/* Matrix[i, j] = Matrix[i, j] >= scalar */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GEQ_SCALAR,

	/* Matrix[i, j] = Matrix[i, j] <= scalar */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LEQ_SCALAR,

	/* Matrix[i, j] = Matrix[i, j] > scalar */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GT_SCALAR,

	/* Matrix[i, j] = Matrix[i, j] < scalar */
	BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LT_SCALAR
};

class Matrix
{
private:
	int rDim = 0;
	int cDim = 0;
	float *matrix = 0;
	void AddScalar(float scalr);
	void SubtractScalar(float scalr);
	void SubtractFromScalar(float scalr);
	void PowerScalar(float scalr);
	void MatrixPower(float scalr);
	void MultiplyScalar(float scalr);
	void ReciprocalMultiply(float scalr);

public:
	Matrix(int rDim, int cDim);
	Matrix(Matrix &other);
	Matrix(int rDim, int cDim, float *raw_data);
	~Matrix();

	/* Get/Set */
	int numCols() { return this->cDim; }
	int numRows() { return this->rDim; }
	float* getRaw() { return this->matrix; }

	/* Matrix Functions */
	void AddBiasRow();
	void AddBiasCol();
	void operateOnMatrixValues(float scalar, ScalarOps opType);
	void operateOnMatrixValues(float scalar, BooleanOps opType);
	void operateOnMatrixValues(Matrix *otherMatrix, BooleanOps opType);
	void Transpose();
	void Log_e();

	/* Returns a row-vector with the mean of every column, similar to MatLab's mean() command */
	Matrix *Mean();
	/* Returns a row-vector with the standard deviation of every column, similar to MatLab's std() command */
	Matrix *StdDev();
	/* Returns a row-vector with the sum of every column, similar to MatLab's sum() command */
	Matrix *Sum();
	Matrix *MaxRowNumber();

	/* Operators */
	float &operator[] (Indexer *operand);
	const float &operator[] (const Indexer *operand) const;
	float &operator[] (int index);
	Matrix *operator* (const Matrix &operand);
	Matrix *operator+ (const Matrix &operand);
	Matrix *operator- (const Matrix &operand);
	Matrix *operator^ (const Matrix &operand);

	/* Helper Functions */
	static void printMatrix(Matrix *matrixToPrint);
	static Matrix *MakeIdentityMatrix(int dims);
	static Matrix *MakeZeroMatrix(int dims);
	static Matrix *LoadMatrix(std::string fileName, char col_delimiter);
};

#endif /* Matrix_hpp */
