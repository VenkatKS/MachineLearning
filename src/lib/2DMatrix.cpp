//
//  Matrix.cpp
//
//
//  Created by Venkat Srinivasan on 6/25/17.
//
//

#include <sstream>
#include <vector>
#include <math.h>
#include <cassert>
#include "include/2DMatrix.hpp"

bool set = false;
Matrix::Matrix(int rDim, int cDim)
{
	this->rDim = rDim;
	this->cDim = cDim;

	this->matrix = new double[rDim * cDim];

	/* Initialize new Matrix to all 0s */
	memset(this->matrix, 0, sizeof(double) * (rDim * cDim));
}

Matrix::Matrix(Matrix &other)
{
	this->cDim = other.cDim;
	this->rDim = other.rDim;

	this->matrix = new double[rDim * cDim];

	/* Initialize new Matrix to all 0s */
	memcpy(this->matrix, other.matrix, sizeof(double) * (rDim * cDim));
}

Matrix::Matrix(int rDim, int cDim, double *raw_data)
{
	this->cDim = cDim;
	this->rDim = rDim;

	this->matrix = raw_data;
}

double &Matrix::operator[](Indexer *operand)
{
	if (operand->rowID >= this->numRows() || operand->colID >= this->numCols())
		assert(0);

	return (this->matrix[(operand->rowID * this->cDim) + operand->colID]);
}

double &Matrix::operator[](int index)
{
	return (this->matrix[index]);
}

const double &Matrix::operator[] (const Indexer *operand)
const {
	if (operand->rowID >= this->rDim || operand->colID >= this->cDim)
		assert(0);

	return (this->matrix[(operand->rowID * this->cDim) + operand->colID]);
}

Matrix *Matrix::operator*(const Matrix &operand)
{
	/* Matrix crawling iterators */
	int op1_r_idx = 0;
	int op1_c_idx = 0;

	int op2_c_idx = 0;

	/* Ensure that the dimensions are proper */
	if (this->cDim != operand.rDim) return NULL;

	Matrix *resultant = new Matrix(this->rDim, operand.cDim);

	for (op1_r_idx = 0; op1_r_idx < this->rDim; op1_r_idx++)
	{
		for (op2_c_idx = 0; op2_c_idx < operand.cDim; op2_c_idx++)
		{
			double runningSum = 0;
			for (op1_c_idx = 0; op1_c_idx < this->cDim; op1_c_idx++)
			{
				Indexer *op1 = new Indexer(op1_r_idx, op1_c_idx);
				Indexer *op2 = new Indexer(op1_c_idx, op2_c_idx);

				runningSum += (*this)[op1] * operand[op2];

				delete op1;
				delete op2;
			}
			Indexer *opIdx = new Indexer(op1_r_idx, op2_c_idx);
			(*resultant)[opIdx] = runningSum;
			delete opIdx;
		}
	}

	return resultant;
}

Matrix *Matrix::operator+(const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;

	int idx = 0;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		resultant->matrix[idx] = this->matrix[idx] + operand.matrix[idx];
	}

	return resultant;
}

Matrix *Matrix::operator-(const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;

	int idx = 0;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		resultant->matrix[idx] = this->matrix[idx] - operand.matrix[idx];
	}

	return resultant;
}

Matrix *Matrix::operator^ (const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;

	int idx = 0;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		resultant->matrix[idx] = pow (this->matrix[idx], operand.matrix[idx]);
	}

	return resultant;
}

void Matrix::PowerScalar(double scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = pow(this->matrix[idx], scalr);
	}
}

void Matrix::MatrixPower(double scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = pow(scalr, this->matrix[idx]);
	}
}

void Matrix::AddScalar(double scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] + scalr;
	}
}

void Matrix::SubtractScalar(double scalr)
{
	if (scalr < 0) printf("WARNING_SubtractScalar: scalr should be positive for subtraction, negative for addition.");

	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] - scalr;
	}
}

void Matrix::SubtractFromScalar(double scalr)
{
	if (scalr < 0) printf("WARNING_SubtractScalar: scalr should be positive for subtraction, negative for addition.");

	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = scalr - this->matrix[idx];
	}
}

void Matrix::operateOnMatrixValues(double scalar, ScalarOps opType)
{
	switch (opType)
	{
		case OP_ADD_SCALAR_TO_EVERY_MATRIX_ELEMENT:
			AddScalar(scalar);
			return;
		case OP_SUBTRACT_SCALAR_FROM_EVERY_MATRIX_ELEMENT:
			SubtractScalar(scalar);
			return;
		case OP_MULTIPLY_SCALAR_WITH_EVERY_MATRIX_ELEMENT:
			MultiplyScalar(scalar);
			return;
		case OP_RAISE_EVERY_MATRIX_ELEMENT_TO_SCALAR_POWER:
			PowerScalar(scalar);
			return;
		case OP_SUBTRACT_EVERY_MATRIX_ELEMENT_FROM_SCALAR:
			SubtractFromScalar(scalar);
			return;
		case OP_RAISE_SCALAR_TO_EVERY_MATRIX_ELEMENT_POWER:
			MatrixPower(scalar);
			return;
		case OP_INVERT_EVERY_MATRIX_ELEMENT_AND_MULTIPLY_SCALAR:
			ReciprocalMultiply(scalar);
			return;
		default:
			assert (0);
			break;
	}
}

void Matrix::operateOnMatrixValues(double scalar, BooleanOps opType)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		double this_element = this->matrix[idx];
		bool result = 0;

		switch (opType)
		{
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR:
				result = (this_element == scalar);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_NOT_EQUAL_TO_SCALAR:
				result = (this_element != scalar);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GEQ_SCALAR:
				result = (this_element >= scalar);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LEQ_SCALAR:
				result = (this_element <= scalar);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GT_SCALAR:
				result = (this_element > scalar);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LT_SCALAR:
				result = (this_element < scalar);
				break;
			default:
				assert (0);
				break;
		}

		this->matrix[idx] = (double) result;
	}
}

void Matrix::operateOnMatrixValues(Matrix *otherMatrix, BooleanOps opType)
{
	int idx = 0;

	if (this->numCols() != otherMatrix->numCols() || this->numRows() != otherMatrix->numRows())
		return;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		double this_element = this->matrix[idx];
		double that_element = otherMatrix->matrix[idx];
		bool result = 0;

		switch (opType)
		{
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR:
				result = (this_element == that_element);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_NOT_EQUAL_TO_SCALAR:
				result = (this_element != that_element);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GEQ_SCALAR:
				result = (this_element >= that_element);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LEQ_SCALAR:
				result = (this_element <= that_element);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_GT_SCALAR:
				result = (this_element > that_element);
				break;
			case BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_LT_SCALAR:
				result = (this_element < that_element);
				break;
			default:
				assert (0);
				break;
		}

		this->matrix[idx] = (double) result;
	}
}

void Matrix::Log_e()
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = log(this->matrix[idx]);
	}
}

void Matrix::MultiplyScalar(double scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] * scalr;
	}
}

void Matrix::ReciprocalMultiply(double scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = (double)(scalr/ ((double)this->matrix[idx]));
	}
}

Matrix *appendMatrix (const Matrix &other)
{

	return NULL;
}

void Matrix::printMatrix(Matrix *matrixToPrint)
{
	int rDims = matrixToPrint->rDim;
	int cDims = matrixToPrint->cDim;
	int r_idx = 0;
	int c_idx = 0;

	for (r_idx = 0; r_idx < rDims; r_idx++)
	{
		for (c_idx = 0; c_idx < cDims; c_idx++)
		{
			Indexer *access = new Indexer(r_idx, c_idx);
			printf("%f", (*matrixToPrint)[access]);
			printf(" ");
			delete access;
		}
		printf("\n");
	}
}

Matrix *Matrix::LoadMatrix(std::string fileName, char col_delimiter)
{
	std::ifstream infile(fileName);

	std::string nextLine;

	std::vector<std::vector<double>> lines;

	/* iterators */
	int r_idx = 0;
	int c_idx = 0;

	while (std::getline(infile, nextLine))
	{
		std::istringstream iss(nextLine);
		std::string token;
		std::vector<double> lineTokens = {};

		{
			std::getline(iss, token, col_delimiter);
			double value = std::atof(token.c_str());
			lineTokens.push_back(value);

			while (std::getline(iss, token, col_delimiter))
			{
				value = std::atof(token.c_str());
				lineTokens.push_back(value);
			}
		}

		lines.push_back(lineTokens);
	}

	Matrix *generatedMatrix = new Matrix((int) lines.size(), (int) lines[0].size());

	for (r_idx = 0; r_idx < lines.size(); r_idx++)
	{
		std::vector<double> line = lines[r_idx];
		for (c_idx = 0; c_idx < line.size(); c_idx++)
		{
			Indexer *currentElement = new Indexer(r_idx, c_idx);
			(*generatedMatrix)[currentElement] = line[c_idx];
			delete currentElement;
		}
	}

	return generatedMatrix;
}

void Matrix::AddBiasRow()
{
	int idx = 0;
	double *new_matrix = new double[(this->rDim + 1) * this->cDim];

	for (idx = 0; idx < this->cDim; idx++)
	{
		new_matrix[(0 * this->cDim) + idx] = 1;
	}

	memcpy(&new_matrix[(1 * this->cDim) + 0], this->matrix, sizeof(double) * (this->rDim * this->cDim));

	delete this->matrix;

	this->rDim = this->rDim + 1;

	this->matrix = new_matrix;
}

void Matrix::AddBiasCol()
{
	int idx = 0;
	int r_idx = 0;
	int c_idx = 0;

	double *new_matrix = new double[this->rDim * (this->cDim + 1)];

	for (idx = 0; idx < this->rDim; idx++)
	{
		new_matrix[(idx * (this->cDim + 1)) + 0] = 1;
	}

	for (r_idx = 0; r_idx < this->rDim; r_idx++)
	{
		for (c_idx = 0; c_idx < this->cDim; c_idx++)
		{
			new_matrix[(r_idx * (this->cDim + 1)) + (1 + c_idx)] = this->matrix[(r_idx * this->cDim) + c_idx];
		}
	}

	delete this->matrix;

	this->cDim = this->cDim + 1;

	this->matrix = new_matrix;
}

void Matrix::Transpose()
{
	double *new_matrix = new double[this->rDim * (this->cDim)];
	int r_idx = 0;
	int c_idx = 0;
	int temp = 0;

	for (r_idx = 0; r_idx < this->rDim; r_idx++)
	{
		for (c_idx = 0; c_idx < this->cDim; c_idx++)
		{
			new_matrix[(c_idx * (this->rDim)) + r_idx] = this->matrix[(r_idx * this->cDim) + c_idx];
		}
	}
	temp = this->cDim;
	this->cDim = this->rDim;
	this->rDim = temp;
	delete this->matrix;
	this->matrix = new_matrix;
}

Matrix *Matrix::Mean()
{
	int c_idx, r_idx = 0;
	Matrix &data = (*this);
	Matrix &Data_Mean = *(new Matrix(1, data.numCols()));

	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		double colRunningCount = 0;

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			colRunningCount = colRunningCount + data[currentIndex];
			delete currentIndex;
		}

		Indexer *currentMean = new Indexer(0, c_idx);
		Data_Mean[currentMean] = (((double)(colRunningCount)) / ((double) data.numRows()));
		delete currentMean;
	}
	return &Data_Mean;
}

Matrix *Matrix::StdDev()
{
	int c_idx, r_idx = 0;
	Matrix &data = (*this);
	Matrix &Data_Mean = (*this->Mean());
	Matrix &Data_STD = *(new Matrix(1, data.numCols()));

	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		Indexer *colMeanIndex = new Indexer(0, c_idx);
		double colRunningCount = 0;
		double colMean = Data_Mean[colMeanIndex];

		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);

			double indexValue = data[currentIndex];
			indexValue = indexValue - colMean;
			indexValue = fabs(indexValue);
			indexValue = pow(indexValue, (double) 2);
			colRunningCount = colRunningCount + indexValue;

			delete currentIndex;
		}

		/* NOTE: Uses MATLAB's formula for standard deviation */
		colRunningCount = (((double)(colRunningCount)) / ((double) (data.numRows() - 1)));
		Data_STD[colMeanIndex] = (sqrt(colRunningCount));

		delete colMeanIndex;
	}

	delete &Data_Mean;
	return &Data_STD;
}

Matrix *Matrix::Sum()
{
	int c_idx, r_idx = 0;
	Matrix &data = (*this);
	Matrix &Data_Sum = (*new Matrix(1, data.numCols()));

	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		double runningColCount = 0;
		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			runningColCount = runningColCount + data[currentIndex];
			delete currentIndex;
		}
		Indexer *currentMean = new Indexer(0, c_idx);
		Data_Sum[currentMean] = runningColCount;
		delete currentMean;
	}

	return &Data_Sum;
}

Matrix *Matrix::MaxRowNumber()
{
	int c_idx, r_idx = 0;
	Matrix &data = (*this);
	Matrix &Data_Sum = (*new Matrix(1, data.numCols()));

	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		double currentMax = -__DBL_MAX__;
		int cur_ridx = 0;
		for (r_idx = 0; r_idx < data.numRows(); r_idx++)
		{
			Indexer *currentIndex = new Indexer(r_idx, c_idx);
			if (data[currentIndex] > currentMax)
			{
				currentMax = data[currentIndex];
				cur_ridx = r_idx;
			}
			delete currentIndex;
		}
		Indexer *currentCol = new Indexer(0, c_idx);
		Data_Sum[currentCol] = cur_ridx;
		delete currentCol;
	}

	return &Data_Sum;
}

Matrix::~Matrix()
{
	delete this->matrix;
}
