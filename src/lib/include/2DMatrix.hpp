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

class Matrix
{
private:
	int rDim = 0;
	int cDim = 0;
	double *matrix = 0;
	void AddScalar(double scalr);
	void SubtractScalar(double scalr);
	void SubtractFromScalar(double scalr);
	void PowerScalar(double scalr);
	void MatrixPower(double scalr);
	void MultiplyScalar(double scalr);
	void ReciprocalMultiply(double scalr);

public:
	Matrix(int rDim, int cDim);
	Matrix(Matrix &other);
	Matrix(int rDim, int cDim, double *raw_data);
	~Matrix();

	/* Get/Set */
	int numCols() { return this->cDim; }
	int numRows() { return this->rDim; }
	double* getRaw() { return this->matrix; }

	/* Matrix Functions */
	void AddBiasRow();
	void AddBiasCol();
	void operateOnMatrixValues(double scalar, ScalarOps opType);
	void Transpose();
	void Log_e();

	/* Operators */
	double &operator[] (Indexer *operand);
	const double &operator[] (const Indexer *operand) const;
	double &operator[] (int index);
	Matrix *operator* (const Matrix &operand);
	Matrix *operator- (const Matrix &operand);
	Matrix *operator^ (const Matrix &operand);

	/* Helper Functions */
	static void printMatrix(Matrix *matrixToPrint);
	static Matrix *MakeIdentityMatrix(int dims);
	static Matrix *MakeZeroMatrix(int dims);
	static Matrix *LoadMatrix(std::string fileName);

};

#endif /* Matrix_hpp */
