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
#include "../OpenCL/include/opencl_driver.hpp"

bool set = false;
Matrix::Matrix(int rDim, int cDim)
{
	this->rDim = rDim;
	this->cDim = cDim;

	this->matrix = new float[rDim * cDim];

	/* Initialize new Matrix to all 0s */
	memset(this->matrix, 0, sizeof(float) * (rDim * cDim));
}

Matrix::Matrix(Matrix &other)
{
	this->cDim = other.cDim;
	this->rDim = other.rDim;

	this->matrix = new float[rDim * cDim];

	/* Initialize new Matrix to all 0s */
	memcpy(this->matrix, other.matrix, sizeof(float) * (rDim * cDim));
}

Matrix::Matrix(int rDim, int cDim, float *raw_data)
{
	this->cDim = cDim;
	this->rDim = rDim;

	this->matrix = raw_data;
}

float &Matrix::operator[](Indexer *operand)
{
	if (operand->rowID >= this->numRows() || operand->colID >= this->numCols())
		assert(0);

	return (this->matrix[(operand->rowID * this->cDim) + operand->colID]);
}

float &Matrix::operator[](int index)
{
	return (this->matrix[index]);
}

const float &Matrix::operator[] (const Indexer *operand)
const {
	if (operand->rowID >= this->rDim || operand->colID >= this->cDim)
		assert(0);

	return (this->matrix[(operand->rowID * this->cDim) + operand->colID]);
}

Matrix *Matrix::operator*(const Matrix &operand)
{
	Matrix *result_matrix = new Matrix(this->rDim, operand.cDim);

	/* Ensure that the dimensions are proper */
	if (this->cDim != operand.rDim)
		return NULL;

	size_t localWorkSize[3];
	size_t globalWorkSize[3];
	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = this->rDim;
	globalWorkSize[1] = operand.cDim;
	ml_opencl_execution_state multiply_state =
	{
		2,
		this->matrix,
		(this->rDim * this->cDim),
		operand.matrix,
		(operand.rDim * operand.cDim),
		(this->cDim),
		(operand.cDim),
		(this->rDim),
		result_matrix->matrix,
		result_matrix->rDim * result_matrix->cDim,
		NULL,
		globalWorkSize,
		2
	};

	get_active_session().execute_kernel(multiply_state);
	return result_matrix;
}

Matrix *Matrix::operator+(const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	size_t global = (this->rDim * this->cDim);
	ml_opencl_execution_state sum_state =
	{
		3,
		this->matrix,
		(this->rDim * this->cDim),
		operand.matrix,
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		resultant->matrix,
		resultant->rDim * resultant->cDim,
		NULL,
		&global,
		1
	};

	get_active_session().execute_kernel(sum_state);

	return resultant;
}

Matrix *Matrix::operator-(const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	size_t global = (this->rDim * this->cDim);

	ml_opencl_execution_state minus_state =
	{
		4,
		this->matrix,
		(this->rDim * this->cDim),
		operand.matrix,
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		resultant->matrix,
		resultant->rDim * resultant->cDim,
		NULL,
		&global,
		1
	};

	get_active_session().execute_kernel(minus_state);

	return resultant;
}

Matrix *Matrix::operator^ (const Matrix &operand)
{
	if (this->rDim != operand.rDim || this->cDim != operand.cDim)
		return NULL;
	Matrix *resultant = new Matrix(this->rDim, this->cDim);

	size_t global = (this->rDim * this->cDim);

	ml_opencl_execution_state power_state =
	{
		5,
		this->matrix,
		(this->rDim * this->cDim),
		operand.matrix,
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		(operand.rDim * operand.cDim),
		(this->rDim * this->cDim),
		resultant->matrix,
		resultant->rDim * resultant->cDim,
		NULL,
		&global,
		1
	};

	get_active_session().execute_kernel(power_state);

	return resultant;
}

void Matrix::PowerScalar(float scalr)
{
	size_t global = (this->rDim * this->cDim);
	ml_opencl_execution_state power_scalar_state =
	{
		1,
		this->matrix,
		(this->rDim * this->cDim),
		&scalr,
		1,
		(this->rDim * this->cDim),
		1,
		(this->rDim * this->cDim),
		this->matrix,
		(this->rDim * this->cDim),
		NULL,
		&global,
		1
	};

	get_active_session().execute_kernel(power_scalar_state);
}

void Matrix::MatrixPower(float scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = pow(scalr, this->matrix[idx]);
	}
}

void Matrix::AddScalar(float scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] + scalr;
	}
}

void Matrix::SubtractScalar(float scalr)
{
	if (scalr < 0) printf("WARNING_SubtractScalar: scalr should be positive for subtraction, negative for addition.");

	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] - scalr;
	}
}

void Matrix::SubtractFromScalar(float scalr)
{
	if (scalr < 0) printf("WARNING_SubtractScalar: scalr should be positive for subtraction, negative for addition.");

	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = scalr - this->matrix[idx];
	}
}

void Matrix::operateOnMatrixValues(float scalar, ScalarOps opType)
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

void Matrix::operateOnMatrixValues(float scalar, BooleanOps opType)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		float this_element = this->matrix[idx];
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

		this->matrix[idx] = (float) result;
	}
}

void Matrix::operateOnMatrixValues(Matrix *otherMatrix, BooleanOps opType)
{
	int idx = 0;

	if (this->numCols() != otherMatrix->numCols() || this->numRows() != otherMatrix->numRows())
		return;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		float this_element = this->matrix[idx];
		float that_element = otherMatrix->matrix[idx];
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

		this->matrix[idx] = (float) result;
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

void Matrix::MultiplyScalar(float scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = this->matrix[idx] * scalr;
	}
}

void Matrix::ReciprocalMultiply(float scalr)
{
	int idx = 0;

	for (idx = 0; idx < (this->rDim * this->cDim); idx++)
	{
		this->matrix[idx] = (float)(scalr/ ((float)this->matrix[idx]));
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

	if (infile.fail())
	{
		printf("Specified file (%s) cannot be opened.\n", fileName.c_str());
		return NULL;
	}
	std::string nextLine;

	std::vector<std::vector<float>> lines;

	/* iterators */
	int r_idx = 0;
	int c_idx = 0;

	while (std::getline(infile, nextLine))
	{
		std::istringstream iss(nextLine);
		std::string token;
		std::vector<float> lineTokens = {};

		{
			std::getline(iss, token, col_delimiter);
			float value = std::atof(token.c_str());
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
		std::vector<float> line = lines[r_idx];
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
	float *new_matrix = new float[(this->rDim + 1) * this->cDim];

	for (idx = 0; idx < this->cDim; idx++)
	{
		new_matrix[(0 * this->cDim) + idx] = 1;
	}

	memcpy(&new_matrix[(1 * this->cDim) + 0], this->matrix, sizeof(float) * (this->rDim * this->cDim));

	delete this->matrix;

	this->rDim = this->rDim + 1;

	this->matrix = new_matrix;
}

void Matrix::AddBiasCol()
{
	int idx = 0;
	int r_idx = 0;
	int c_idx = 0;

	float *new_matrix = new float[this->rDim * (this->cDim + 1)];

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
	float *new_matrix = new float[this->rDim * (this->cDim)];
	int temp = 0;

	size_t global[2];
	global[0] = (this->numRows());
	global[1] = (this->numCols());
	ml_opencl_execution_state transpose_state =
	{
		9,
		this->matrix,
		this->cDim * this->rDim,
		NULL,
		1,
		this->rDim,
		this->cDim,
		0,
		new_matrix,
		this->cDim * this->rDim,
		NULL,
		global,
		2
	};
	get_active_session().execute_kernel(transpose_state);

	temp = this->cDim;
	this->cDim = this->rDim;
	this->rDim = temp;
	delete this->matrix;
	this->matrix = new_matrix;
}

Matrix *Matrix::Mean()
{
	static float time_spent_here = 0;
	const clock_t begin_time = clock();

	Matrix &data = (*this);
	Matrix &Data_Mean = *(new Matrix(1, data.numCols()));

	size_t global[1];
	global[0] = (data.numCols());
	ml_opencl_execution_state mean_state =
	{
		6,
		this->matrix,
		this->cDim * this->rDim,
		NULL,
		1,
		this->rDim,
		this->cDim,
		0,
		Data_Mean.matrix,
		1 * data.cDim,
		NULL,
		global,
		1
	};
	get_active_session().execute_kernel(mean_state);
	time_spent_here += float( clock () - begin_time ) /  CLOCKS_PER_SEC;

	return &Data_Mean;
}

Matrix *Matrix::StdDev()
{

	Matrix &data = (*this);
	Matrix &Data_Mean = (*this->Mean());
	Matrix &Data_STD = *(new Matrix(1, data.numCols()));

	size_t global[1];
	global[0] = (data.numCols());
	ml_opencl_execution_state mean_state =
	{
		8,
		this->matrix,
		this->cDim * this->rDim,
		Data_Mean.matrix,
		Data_Mean.cDim * Data_Mean.rDim,
		this->rDim,
		this->cDim,
		0,
		Data_STD.matrix,
		1 * data.cDim,
		NULL,
		global,
		1
	};
	get_active_session().execute_kernel(mean_state);

	delete &Data_Mean;
	return &Data_STD;
}

Matrix *Matrix::Sum()
{
	Matrix &data = (*this);
	Matrix &Data_Sum = (*new Matrix(1, data.numCols()));

	size_t global[1];
	global[0] = (data.numCols());
	ml_opencl_execution_state sum_state =
	{
		7,
		this->matrix,
		this->cDim * this->rDim,
		NULL,
		1,
		this->rDim,
		this->cDim,
		0,
		Data_Sum.matrix,
		1 * data.cDim,
		NULL,
		global,
		1
	};
	get_active_session().execute_kernel(sum_state);
	return &Data_Sum;
}

Matrix *Matrix::MaxRowNumber()
{
	int c_idx, r_idx = 0;
	Matrix &data = (*this);
	Matrix &Data_Sum = (*new Matrix(1, data.numCols()));

	for (c_idx = 0; c_idx < data.numCols(); c_idx++)
	{
		float currentMax = -__DBL_MAX__;
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
