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

class Matrix
{
	int rDim = 0;
	int cDim = 0;
	double *matrix = 0;

public:
	Matrix(int rDim, int cDim);
	Matrix(Matrix &other);
	~Matrix();

	/* Get/Set */
	int numCols() { return this->cDim; }
	int numRows() { return this->rDim; }

	/* Matrix Functions */
	void AddBiasRow();
	void AddBiasCol();
	void AddScalar(double scalr);
	void SubtractScalar(double scalr);
	void SubtractFromScalar(double scalr);
	void PowerScalar(double scalr);
	void MatrixPower(double scalr);
	void MultiplyScalar(double scalr);
	void Log_e();
	void ReciprocalMultiply(double scalr);
	void Transpose();

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
