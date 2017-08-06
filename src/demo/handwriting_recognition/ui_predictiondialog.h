/********************************************************************************
** Form generated from reading UI file 'predictiondialog.ui'
**
** Created by: Qt User Interface Compiler version 5.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PREDICTIONDIALOG_H
#define UI_PREDICTIONDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_PredictionDialog
{
public:
    QGraphicsView *imageView;
    QPushButton *pushButton;
    QPushButton *nextButton;
    QLabel *statusLabel;
    QLabel *actualLabel;
    QLabel *predictedLabel;
    QRadioButton *radioButton;
    QLabel *label;
    QRadioButton *radioButton_2;
    QLabel *label_2;
    QLabel *learningRateLabel;
    QSlider *horizontalSlider;
    QLabel *regRateLabel;
    QSlider *horizontalSlider_2;
    QProgressBar *learningBar;
    QLabel *label_5;
    QLabel *label_3;
    QSpinBox *iterationCount;
    QLabel *label_4;

    void setupUi(QDialog *PredictionDialog)
    {
        if (PredictionDialog->objectName().isEmpty())
            PredictionDialog->setObjectName(QStringLiteral("PredictionDialog"));
        PredictionDialog->resize(691, 487);
        imageView = new QGraphicsView(PredictionDialog);
        imageView->setObjectName(QStringLiteral("imageView"));
        imageView->setGeometry(QRect(160, 10, 511, 401));
        pushButton = new QPushButton(PredictionDialog);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(390, 420, 113, 32));
        nextButton = new QPushButton(PredictionDialog);
        nextButton->setObjectName(QStringLiteral("nextButton"));
        nextButton->setEnabled(false);
        nextButton->setGeometry(QRect(560, 420, 113, 32));
        statusLabel = new QLabel(PredictionDialog);
        statusLabel->setObjectName(QStringLiteral("statusLabel"));
        statusLabel->setGeometry(QRect(10, 470, 671, 16));
        actualLabel = new QLabel(PredictionDialog);
        actualLabel->setObjectName(QStringLiteral("actualLabel"));
        actualLabel->setGeometry(QRect(150, 420, 60, 16));
        predictedLabel = new QLabel(PredictionDialog);
        predictedLabel->setObjectName(QStringLiteral("predictedLabel"));
        predictedLabel->setGeometry(QRect(150, 440, 241, 21));
        radioButton = new QRadioButton(PredictionDialog);
        radioButton->setObjectName(QStringLiteral("radioButton"));
        radioButton->setGeometry(QRect(10, 250, 141, 21));
        radioButton->setChecked(true);
        label = new QLabel(PredictionDialog);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(50, 230, 60, 16));
        radioButton_2 = new QRadioButton(PredictionDialog);
        radioButton_2->setObjectName(QStringLiteral("radioButton_2"));
        radioButton_2->setGeometry(QRect(10, 270, 131, 21));
        radioButton_2->setCheckable(false);
        label_2 = new QLabel(PredictionDialog);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(40, 10, 81, 16));
        learningRateLabel = new QLabel(PredictionDialog);
        learningRateLabel->setObjectName(QStringLiteral("learningRateLabel"));
        learningRateLabel->setGeometry(QRect(10, 30, 141, 21));
        horizontalSlider = new QSlider(PredictionDialog);
        horizontalSlider->setObjectName(QStringLiteral("horizontalSlider"));
        horizontalSlider->setGeometry(QRect(40, 50, 101, 16));
        horizontalSlider->setOrientation(Qt::Horizontal);
        regRateLabel = new QLabel(PredictionDialog);
        regRateLabel->setObjectName(QStringLiteral("regRateLabel"));
        regRateLabel->setGeometry(QRect(10, 70, 141, 16));
        horizontalSlider_2 = new QSlider(PredictionDialog);
        horizontalSlider_2->setObjectName(QStringLiteral("horizontalSlider_2"));
        horizontalSlider_2->setGeometry(QRect(40, 90, 101, 16));
        horizontalSlider_2->setOrientation(Qt::Horizontal);
        learningBar = new QProgressBar(PredictionDialog);
        learningBar->setObjectName(QStringLiteral("learningBar"));
        learningBar->setGeometry(QRect(560, 460, 118, 23));
        learningBar->setValue(0);
        label_5 = new QLabel(PredictionDialog);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(439, 460, 121, 20));
        label_3 = new QLabel(PredictionDialog);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(10, 110, 171, 21));
        iterationCount = new QSpinBox(PredictionDialog);
        iterationCount->setObjectName(QStringLiteral("iterationCount"));
        iterationCount->setGeometry(QRect(80, 130, 48, 24));
        iterationCount->setMinimum(1);
        iterationCount->setMaximum(200);
        iterationCount->setValue(50);
        label_4 = new QLabel(PredictionDialog);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(10, 130, 71, 21));

        retranslateUi(PredictionDialog);

        QMetaObject::connectSlotsByName(PredictionDialog);
    } // setupUi

    void retranslateUi(QDialog *PredictionDialog)
    {
        PredictionDialog->setWindowTitle(QApplication::translate("PredictionDialog", "Venkat Machine Learning", Q_NULLPTR));
        pushButton->setText(QApplication::translate("PredictionDialog", "Learn", Q_NULLPTR));
        nextButton->setText(QApplication::translate("PredictionDialog", "Next", Q_NULLPTR));
        statusLabel->setText(QApplication::translate("PredictionDialog", "Status: Idle", Q_NULLPTR));
        actualLabel->setText(QApplication::translate("PredictionDialog", "Actual:", Q_NULLPTR));
        predictedLabel->setText(QApplication::translate("PredictionDialog", "Predicted:", Q_NULLPTR));
        radioButton->setText(QApplication::translate("PredictionDialog", "Logistic Regression", Q_NULLPTR));
        label->setText(QApplication::translate("PredictionDialog", "Algorithm", Q_NULLPTR));
        radioButton_2->setText(QApplication::translate("PredictionDialog", "Neural Networks", Q_NULLPTR));
        label_2->setText(QApplication::translate("PredictionDialog", "Parameters:", Q_NULLPTR));
        learningRateLabel->setText(QApplication::translate("PredictionDialog", "Learning Rate: 0.01", Q_NULLPTR));
        regRateLabel->setText(QApplication::translate("PredictionDialog", "Regularization: 0.01", Q_NULLPTR));
        label_5->setText(QApplication::translate("PredictionDialog", "Learning Progress:", Q_NULLPTR));
        label_3->setText(QApplication::translate("PredictionDialog", "Gradient Descent:", Q_NULLPTR));
        label_4->setText(QApplication::translate("PredictionDialog", "Iterations:", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class PredictionDialog: public Ui_PredictionDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PREDICTIONDIALOG_H
