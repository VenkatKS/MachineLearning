#ifndef PREDICTIONDIALOG_H
#define PREDICTIONDIALOG_H

#include <QDialog>
#include <QThread>
#include <QGraphicsView>
#include <QLabel>
#include <QRunnable>
#include <QProgressBar>
#include <nML/2DMatrix.hpp>
#include <QPushButton>

namespace Ui {
class PredictionDialog;
}

class InformationPackage {
public:
    unsigned char** imgs;
    unsigned char* labels;
    int numImages;
    int numLabels;
    int imgSize;
    int imgrows;
    int imgcols;

    Matrix *All_Matrix;
    Matrix *All_Solutions;
    Matrix *predicted_results;
};

class LearningThread : public QRunnable {
public:
    InformationPackage *deliverables;
    QPushButton *enableButton;
    QGraphicsView *graphicsView;
    QString *imgFileName;
    QString *labelFileName;
    QLabel *actualLabel;
    QLabel *predictedLabel;
    QLabel *statusLabel;
    QProgressBar *progress;

    LearningThread(InformationPackage *deliver, QPushButton *enable, QGraphicsView *view, QString *imgFile, QString *labelFile, QLabel *actual, QLabel *predicted, QLabel *status, QProgressBar *progressBar)
    {
        this->enableButton = enable;
        this->deliverables = deliver;
        this->graphicsView = view;
        this->imgFileName = new QString(*imgFile);
        this->labelFileName = new QString(*labelFile);
        this->actualLabel = actual;
        this->predictedLabel = predicted;
        this->statusLabel = status;
        this->progress = progressBar;
    }

    void run();
};

class PredictionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PredictionDialog(QWidget *parent = 0);
    ~PredictionDialog();

private slots:
    void on_pushButton_clicked();

    void on_nextButton_clicked();

    void on_horizontalSlider_sliderMoved(int position);

    void on_horizontalSlider_2_sliderMoved(int position);

    void on_iterationCount_valueChanged(int arg1);

private:
    Ui::PredictionDialog *ui;
};

#endif // PREDICTIONDIALOG_H
