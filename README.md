# MachineLearning

This repository contains a simple machine learning library with a collection of the most commonly used machine learning algorithms. This is a work in progress and is not completed.

## Linear Algebra Library
The Linear Algebra part of the library is the basis for everything else. It's very straight forward, as it only implements the necessary matricies and matrix operations. The full matrix struct definition file can be found under 2DMatrix.hpp, but here is the recap of what the linear algebra library can do:
### General Notes About Matrices
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
### Matrix Arithmetic Operations
## Regression Library
## Classification Library
## Neural Networks Library
