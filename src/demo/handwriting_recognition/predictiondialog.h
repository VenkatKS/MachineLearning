#ifndef PREDICTIONDIALOG_H
#define PREDICTIONDIALOG_H

#include <QDialog>

namespace Ui {
class PredictionDialog;
}

class PredictionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PredictionDialog(QWidget *parent = 0);
    ~PredictionDialog();

private slots:
    void on_pushButton_clicked();

    void on_nextButton_clicked();

private:
    Ui::PredictionDialog *ui;
};

#endif // PREDICTIONDIALOG_H
