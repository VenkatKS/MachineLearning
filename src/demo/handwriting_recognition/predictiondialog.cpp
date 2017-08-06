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
#include <QThreadPool>
#include <QProgressBar>
#include "predictiondialog.h"
#include "ui_predictiondialog.h"
#include <QSlider>
#include <QSpinBox>

static bool learned = false;
static int idx = 0;

static float learningRate = 0.01;
static float regularizationRate = 0.01;
static float iterations = 50;

LearningThread *workerThread;
InformationPackage *ImageDeliverables;

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

void LearningThread::run() {
     this->deliverables->imgs = read_mnist_images(imgFileName->toUtf8().constData(), this->deliverables->numImages, this->deliverables->imgSize, this->deliverables->imgrows, this->deliverables->imgcols);
     this->deliverables->labels = read_mnist_labels(labelFileName->toUtf8().constData(), this->deliverables->numLabels);

    int numImages = this->deliverables->numImages;
    int imgSize = this->deliverables->imgSize;

     this->deliverables->All_Matrix = new Matrix(this->deliverables->numImages, this->deliverables->imgSize);
     this->deliverables->All_Solutions = new Matrix(numImages, 1);
     for (int i = 0; i < numImages; i++)
     {
         for (int j = 0; j < imgSize; j++)
         {
             Indexer *myIndex = new Indexer(i, j);
             (*this->deliverables->All_Matrix)[myIndex] = (float) (this->deliverables->imgs[i])[j];
             delete myIndex;
         }
     }
     for (int i = 0; i < numImages; i++)
     {
         (*this->deliverables->All_Solutions)[i] = (float) (this->deliverables->labels[i]);
     }
    progress->setValue(20);
     /* Create our machine learning object using the loaded data as our operating data set */
     DataSetWrapper *test_wrapper = new DataSetWrapper(this->deliverables->All_Matrix, this->deliverables->All_Solutions);

     /* Create a linear regression fit model */
     progress->setValue(30);
     LogisiticClassificationFit *logfit = new LogisiticClassificationFit(test_wrapper, 10, learningRate, regularizationRate);
     MachineLearning *logisticOperations = new MachineLearning(*logfit);
     ML_SingleLogOps *multi_ops = (ML_SingleLogOps *) logisticOperations->Algorithms();
     progress->setValue(35);
     Matrix *all_theta = multi_ops->OneVsAll(iterations);
     progress->setValue(40);
     this->deliverables->predicted_results = multi_ops->PredictOneVsAll(*all_theta);
     progress->setValue(60);
     this->deliverables->predicted_results->Transpose();
     progress->setValue(70);
     Matrix *my_results = new Matrix(*this->deliverables->predicted_results);
     my_results->operateOnMatrixValues(this->deliverables->All_Solutions, BOOLEAN_OP_IS_EVERY_MATRIX_ELEMENT_EQUAL_TO_SCALAR);
     progress->setValue(80);
     Matrix *result = my_results->Mean();
     assert (result->numCols() == 1);
     assert (result->numRows() == 1);

     const QImage *newImage = new QImage((uchar*)this->deliverables->imgs[idx], this->deliverables->imgcols, this->deliverables->imgrows, this->deliverables->imgcols, QImage::Format_Indexed8);

     QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*newImage));
     QGraphicsScene* scene = new QGraphicsScene();
     scene->addItem(item);
     graphicsView->setScene(scene);
     graphicsView->show();
     graphicsView->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
     char predicted[30];
     char actual[30];
     char accuracy[30];

     sprintf(predicted, "Actual: %d", this->deliverables->labels[idx]);
     actualLabel->setText(QString(predicted));
     sprintf(accuracy, "Overall Accuracy: %f (%d out of %d)", (*result)[0], (int) ((*result)[0] * numImages), numImages);
     statusLabel->setText(accuracy);
     sprintf(actual, "Predicted: %f", (*this->deliverables->predicted_results)[idx]);
     predictedLabel->setText(QString(actual));
     progress->setValue(100);

     enableButton->setEnabled(true);
     return;
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
    QProgressBar *progress = this->findChild<QProgressBar*>("learningBar");
    QSlider *regRateSlider = this->findChild<QSlider*>("horizontalSlider");
    QSlider *learningRateSlider = this->findChild<QSlider*>("horizontalSlider_2");
    QSpinBox *iterationCount = this->findChild<QSpinBox*>("iterationCount");

    QMessageBox messageBox;

    ImageDeliverables = new InformationPackage();

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
    regRateSlider->setEnabled(false);
    learningRateSlider->setEnabled(false);
    iterationCount->setEnabled(false);
    statusLabel->setText("Currently studying 10,000 images....");
    progress->setValue(0);
    workerThread = new LearningThread(ImageDeliverables, enableButton, graphicsView, &imgFileName, &labelFileName, actualLabel, predictedLabel, statusLabel, progress);
    QThreadPool::globalInstance()->start(workerThread);

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

    const QImage *newImage = new QImage((uchar*)ImageDeliverables->imgs[idx], ImageDeliverables->imgcols, ImageDeliverables->imgrows, ImageDeliverables->imgcols, QImage::Format_Indexed8);

    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*newImage));
    QGraphicsScene* scene = new QGraphicsScene();

    scene->addItem(item);
    graphicsView->setScene(scene);
    graphicsView->show();
    graphicsView->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);

    char predicted[30];
    char actual[30];

    sprintf(predicted, "Actual: %d", ImageDeliverables->labels[idx]);
    sprintf(actual, "Predicted: %f", (*ImageDeliverables->predicted_results)[idx]);

    actualLabel->setText(QString(predicted));
    predictedLabel->setText(QString(actual));
}

void PredictionDialog::on_horizontalSlider_sliderMoved(int position)
{
    QLabel *rateLabel = this->findChild<QLabel*>("learningRateLabel");
    learningRate = 0.01 * position;
    char actual[30];

    sprintf(actual, "%0.2f", learningRate);

    QString newRate("Learning Rate: ");
    newRate.append(actual);
    rateLabel->setText(newRate);

}

void PredictionDialog::on_horizontalSlider_2_sliderMoved(int position)
{
    QLabel *rateLabel = this->findChild<QLabel*>("regRateLabel");
    regularizationRate = 0.01 * position;
    char actual[30];

    sprintf(actual, "%0.2f", regularizationRate);

    QString newRate("Regularization: ");
    newRate.append(actual);
    rateLabel->setText(newRate);
}

void PredictionDialog::on_iterationCount_valueChanged(int arg1)
{
    iterations = arg1;
}
