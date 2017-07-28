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
#include <QtWidgets/QPushButton>

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

    void setupUi(QDialog *PredictionDialog)
    {
        if (PredictionDialog->objectName().isEmpty())
            PredictionDialog->setObjectName(QStringLiteral("PredictionDialog"));
        PredictionDialog->resize(691, 487);
        imageView = new QGraphicsView(PredictionDialog);
        imageView->setObjectName(QStringLiteral("imageView"));
        imageView->setGeometry(QRect(10, 10, 661, 411));
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
    } // retranslateUi

};

namespace Ui {
    class PredictionDialog: public Ui_PredictionDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PREDICTIONDIALOG_H
