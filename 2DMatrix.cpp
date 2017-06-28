//
//  Matrix.cpp
//  
//
//  Created by Venkat Srinivasan on 6/25/17.
//
//

#include "2DMatrix.hpp"
#include <sstream>
#include <vector>
#include <math.h>

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

double &Matrix::operator[](Indexer *operand)
{
	return (this->matrix[(operand->rowID * this->cDim) + operand->colID]);
}

double &Matrix::operator[](int index)
{
	return (this->matrix[index]);
}

const double &Matrix::operator[] (const Indexer *operand)
const {
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
			int runningSum = 0;
			for (op1_c_idx = 0; op1_c_idx < this->cDim; op1_c_idx++)
			{
				Indexer *op1 = new Indexer(op1_r_idx, op1_c_idx);
				Indexer *op2 = new Indexer(op1_c_idx, op2_c_idx);

				runningSum += (*this)[op1] * operand[op2];
			}
			Indexer *opIdx = new Indexer(op1_r_idx, op2_c_idx);
			(*resultant)[opIdx] = runningSum;
		}
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

void Matrix::SubtractScalar(double scalr)
{
	if (scalr < 0) printf("WARNING_SubtractScalar: scalr should be positive for subtraction, negative for addition.");

	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] - scalr;
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
		}
		printf("\n");
	}
}

Matrix *Matrix::LoadMatrix(std::string fileName)
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
			std::getline(iss, token, ',');
			double value = std::atof(token.c_str());
			lineTokens.push_back(value);

			while (std::getline(iss, token, ','))
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

Matrix::~Matrix()
{
	delete this->matrix;
}
