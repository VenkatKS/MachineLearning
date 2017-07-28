#include "predictiondialog.h"
#include "ui_predictiondialog.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <nML/2DMatrix.hpp>
#include <nML/MachineLearningLibrary.hpp>
#include <nML/ml_classification.hpp>
#include <QLabel>
#include <QGraphicsPixmapItem>
#include <QDir>
#include <QMessageBox>
#include <QFileInfo>

static bool learned = false;
static int idx = 0;
static Matrix *All_Matrix;
static Matrix *All_Solutions;
static Matrix *temp_y;
static unsigned char** imgs;
static unsigned char* labels;
int numImages;
int numLabels;
int imgSize;
int imgrows;
int imgcols;

unsigned char** read_mnist_images(std::string full_path, int& number_of_images, int& image_size, int& img_rows, int& img_cols) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);
    std::cout << full_path;

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) assert (0);

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;
        img_rows = n_rows;
        img_cols = n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        assert (0);
    }
}

unsigned char* read_mnist_labels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) assert (0);

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        assert (0);
    }
}


PredictionDialog::PredictionDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PredictionDialog)
{
    ui->setupUi(this);
}

PredictionDialog::~PredictionDialog()
{
    delete ui;
}

void PredictionDialog::on_pushButton_clicked()
{
    QPushButton *enableButton = this->findChild<QPushButton*>("nextButton");
    QLabel *statusLabel = this->findChild<QLabel*>("statusLabel");
    QLabel *actualLabel = this->findChild<QLabel*>("actualLabel");
    QLabel *predictedLabel = this->findChild<QLabel*>("predictedLabel");
    QPushButton *pushButton = this->findChild<QPushButton*>("pushButton");
    QMessageBox messageBox;

    QGraphicsView *graphicsView = this->findChild<QGraphicsView*>("imageView");

    QString defaultPath(QDir::currentPath());
    QString trainingDataPath(defaultPath);
    trainingDataPath.append("/trainingFeatureData.dat");

    QString trainingSolutionPath(defaultPath);
    trainingSolutionPath.append("/trainingSolutionData.dat");

    QString imgFileName(trainingDataPath);
    QString labelFileName(trainingSolutionPath);

    if (!(QFileInfo(trainingDataPath).exists()))
    {
        QString illegalMessage("Status: Illegal ");
        illegalMessage.append(trainingDataPath);
        illegalMessage.append(" file.");
        statusLabel->setText(illegalMessage);
        messageBox.setText("Illegal file");
        messageBox.setInformativeText("Please ensure 'trainingFeatureData.dat' is in current working directory.");
        int ret = messageBox.exec();
        return;
    }

    if (!(QFileInfo(trainingSolutionPath).exists()))
    {
        QString illegalMessage("Status: Illegal ");
        illegalMessage.append(trainingSolutionPath);
        illegalMessage.append(" file.");

        statusLabel->setText(illegalMessage);
        messageBox.setText("Illegal file");
        messageBox.setInformativeText("Please ensure 'trainingSolutionData.dat' is in current working directory.");
        int ret = messageBox.exec();
        return;
    }

    statusLabel->setText("Currently studying 10,000 images....");
    enableButton->setEnabled(true);

    imgs = read_mnist_images(imgFileName.toUtf8().constData(), numImages, imgSize, imgrows, imgcols);
    labels = read_mnist_labels(labelFileName.toUtf8().constData(), numLabels);

    All_Matrix = new Matrix(numImages, imgSize);
    All_Solutions = new Matrix(numImages, 1);
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < imgSize; j++)
        {
            Indexer *myIndex = new Indexer(i, j);
            (*All_Matrix)[myIndex] = (float) (imgs[i])[j];
            delete myIndex;
        }
    }
    for (int i = 0; i < numImages; i++)
    {
        (*All_Solutions)[i] = (float) (labels[i]);
    }

    /* Create our machine learning object using the loaded data as our operating data set */
    DataSetWrapper *test_wrapper = new DataSetWrapper(All_Matrix, All_Solutions);

    /* Create a linear regression fit model */
    LogisiticClassificationFit *logfit = new LogisiticClassificationFit(test_wrapper, 10, 0.01, 0.01);
    MachineLearning *logisticOperations = new MachineLearning(*logfit);
    ML_SingleLogOps *multi_ops = (ML_SingleLogOps *) logisticOperations->Algorithms();
    Matrix *all_theta = multi_ops->OneVsAll(50);
    temp_y = multi_ops->PredictOneVsAll(*all_theta);
    temp_y->Transpose();
    Matrix *my_results = new Matrix(*temp_y);
    my_results->operateOnMatrixValues(All_Solutions, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
    Matrix *result = my_results->Mean();
    assert (result->numCols() == 1);
    assert (result->numRows() == 1);

    const QImage *newImage = new QImage((uchar*)imgs[idx], imgcols, imgrows, imgcols, QImage::Format_Indexed8);

    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*newImage));
    QGraphicsScene* scene = new QGraphicsScene();
    scene->addItem(item);
    graphicsView->setScene(scene);
    graphicsView->show();
    graphicsView->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
    char predicted[30];
    char actual[30];
    char accuracy[30];

    sprintf(predicted, "Actual: %d", labels[idx]);
    sprintf(actual, "Predicted: %f", (*temp_y)[idx]);
    sprintf(accuracy, "Overall Accuracy: %f (%d out of %d)", (*result)[0], (int) ((*result)[0] * numImages), numImages);
    actualLabel->setText(QString(predicted));
    predictedLabel->setText(QString(actual));
    statusLabel->setText(accuracy);

    pushButton->setEnabled(false);
}

void PredictionDialog::on_nextButton_clicked()
{
    idx = idx + 1;

    QPushButton *enableButton = this->findChild<QPushButton*>("nextButton");
    QLabel *statusLabel = this->findChild<QLabel*>("statusLabel");
    QLabel *actualLabel = this->findChild<QLabel*>("actualLabel");
    QLabel *predictedLabel = this->findChild<QLabel*>("predictedLabel");
    QGraphicsView *graphicsView = this->findChild<QGraphicsView*>("imageView");

    const QImage *newImage = new QImage((uchar*)imgs[idx], imgcols, imgrows, imgcols, QImage::Format_Indexed8);

    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*newImage));
    QGraphicsScene* scene = new QGraphicsScene();

    scene->addItem(item);
    graphicsView->setScene(scene);
    graphicsView->show();
    graphicsView->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);

    char predicted[30];
    char actual[30];

    sprintf(predicted, "Actual: %d", labels[idx]);
    sprintf(actual, "Predicted: %f", (*temp_y)[idx]);

    actualLabel->setText(QString(predicted));
    predictedLabel->setText(QString(actual));
}
