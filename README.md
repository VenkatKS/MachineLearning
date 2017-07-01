# MachineLearning

This repository contains a simple machine learning library with a collection of the most commonly used machine learning algorithms. This is a work in progress and is not completed.

## Linear Algebra Library
This part of the library contains the basis of the entire library. It contains code to implement matrices and their associated functions. Here's the structure of a simple matrix:

```C++
class Matrix
{
	int rDim = 0;
	int cDim = 0;
	double *matrix = 0;

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

	/* Matrix Operations */
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
```
## Regression Library
## Classification Library
## Neural Networks Library
