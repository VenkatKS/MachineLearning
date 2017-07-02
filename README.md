# MachineLearning

This repository contains a simple machine learning library with a collection of the most commonly used machine learning algorithms. The library contains support of r This is a work in progress and is not completed.

## Linear Algebra Library
The Linear Algebra part of the library is the basis for everything else. It's very straight forward, as it only implements the necessary matricies and matrix operations. The full matrix struct definition file can be found under 2DMatrix.hpp, but here is the recap of what the linear algebra library can do:
### Accessing a Matrix
The Matrix struct consists of a 2D matrix stored in row-major order within a 1D double-precision array. This 1D array is abstracted out, so you usually won't need to interact with it. Instead you interact with the matrix with the use of the Indexer class:
```C++
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
...
Matrix &exampleMatrix = *(new Matrix(1, 4));
Indexer *currentIndex = new Indexer(0, c_idx);
exampleMatrix[currentIndex] = 0;
```
The general indexer class allows you to very easily index into the underlying 1D array using 2D matrix dimensions. If you really want to access the raw array indexes (and do the array row-major calculations yourself), you can also pass in an integer into the offset operator to index the array yourself and not worry about using the Indexer class.
### Matrix Arithmetic and Boolean Operations
The general function you use to operate on any matrix is the operateOnMatrixValues(...) function. It allows you to perform a variety of different operations on every element in the matrix. The function's prototype is defined as such:
```C++
	void operateOnMatrixValues(double scalar, ScalarOps opType);
	void operateOnMatrixValues(double scalar, BooleanOps opType);
	void operateOnMatrixValues(Matrix *otherMatrix, BooleanOps opType);
```
The functions, depending on what you want to do, take in a scalar (or another matrix) and the type of operation you want to perform as an enumerator value.
```C++
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
```
Calling the function with these enumerator values will perform the operations specified in the comments above each enum value to every element in the matrix independently. If using the function that takes in another matrix, every corresponding element in the provided matrix will be used to operate on the first matrix's values (depending on the type of op specified by the enum value). Obviously, this means that the two matrices must have the same dimensions -- if not, no operation will be performed. More operations will be added here as time goes on, but it's pretty straightforward to add your own if needed.

The boolean operations will set the resultant matrix's values to be 0 or 1, depending on the value in the original matrix and the scalar.
### Other General Matrix Operations
There are other general operations that can be performed on a matrix as well. These include transposing a matrix, natural logging every element in the matrix, and getting the mean/standard deviation of every column in the matrix. These can be done using the following functions:
```C++
void Transpose();
void Log_e();

/* Returns a row-vector with the means of each col */
Matrix *Mean();
/* Returns a row-vector with the standard deviations of each col */
Matrix *StdDev();

```
Transposing and logging the matrix will do the action in place, while getting the mean/std.dev. of each row will return a row-vector with the respective values.
### Matrix To Matrix Operations
Simple actions like matrix multiplication can be done using the standard overloaded operators:
```C++
	Matrix &a = (*new Matrix(1, 2));
	Matrix &b = (*new Matrix(2, 3));
	Matrix &c = (* new Matrix(1, 2));
	...
	/* Multiplying Matrices */
	Matrix *mult = a * b; /* Will return a 1x3 Matrix */
	Matrix *mult2 = b * a; /* Will return a NULL since dimensions don't line up */
	...
	/* Subtracting Matricies */
	Matrix add = a - c; /* Will return the difference 1x2 Matrix */
	Matrix add2 = a - b; /* Will return a NULL since dimensions don't match */
	...
	Etc.
```
### Other Data Operations
To make it easy to load data, the library has several static functions:
```C++
static void printMatrix(Matrix *matrixToPrint);
static Matrix *LoadMatrix(std::string fileName);
```
The load matrix function returns a fully loaded matrix from an ascii file, using new-lines as row divisors and commas as column divisors. Will add support for MATLAB file versions soon.
## Regression Library
The regression library has several features that make it easy to optimize and compute the cost for regression problems.
```C++
	static double computeCost(Matrix &training_X, Matrix &training_y, Matrix &training_theta);
	static Matrix *gradientDescent(Matrix &training_X, Matrix &training_y, Matrix &theta, double alpha, int num_iterations);
```
Calling into the cost function with a provided training feature values and their associated results, along with anticipated training factor constants, will provide back the cost of the training factors using the squared cost function. To actually get the optimized factors, you can call into the gradientDescent function with a starting set of factors, along with the provided data and a desired learning rate and a limit on the number of times to run. The gradientDescent function will use the linear gradient descent algorithm to solve for optimum factors and will return it as a Matrix.
## Classification Library
## Neural Networks Library
