#include "predictiondialog.h"
#include <QApplication>

#define INDEX 3303

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    PredictionDialog predDiag;

    predDiag.show();
    return a.exec();
}
