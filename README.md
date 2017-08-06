# MachineLearning

This repository contains a simple machine learning library with a collection of the most commonly used machine learning algorithms. This is a work in progress and is not completed.
## Demos
### Handwriting Recognition Demo
I've developed a quick and dirty handwriting recognition algorithm using this machine learning library and using the MNIST handwritten digits database. The code for this can be found under src/demo/handwriting_recognition. Here are a couple of screenshots:

<p align="center">
<img src="https://github.com/VenkatKS/MachineLearning/blob/master/src/demo/handwriting_recognition/screenshots/digit7.png?raw=true" alt="Digit 7 Prediction" width="50%" height="50%"/>
</p>

<p align="center">
<img src="https://github.com/VenkatKS/MachineLearning/blob/master/src/demo/handwriting_recognition/screenshots/digit4.png?raw=true" alt="Digit 4 Prediction" width="50%" height="50%"/>
</p>
<p>
The algorithm teaches itself the various hand-written digits using the included MNIST database of handwritten digits and then tries to predict a test set. You can start the process by clicking "Learn" (it'll take a while) and then using the "Next" button to scroll through the images included in the MNIST database and see the actual value of those digits (under the label "Actual") and what the algorithm thinks the digit is (under the label "Predicted").
</p>
<p>
The demostration has about a 75% accuracy, and this is mainly because it uses a logistic gradient descent approach of about 50 iterations (alpha of 0.01, and regularization lambda of 0.01). Tweaking these parameters using the demo's sliders and spinboxes to get better results using a cross-verification set and a learning curve will yield significantly better results. As will using the neural network algorithm instead of the logistic regression algorithm. TBD.
</p>

## Progress
* Linear Algebra Library (Done)
  * ~~Fully functional and ground-up Matrix class~~ (Done)
* Polynomial Regression (TBD)
* Linear Regression (Done)
  * ~~Single parameter regression~~ (Done)
  * ~~Multi parameter regression~~ (Done)
* Logistic Classification (Done)
  * ~~Single parameter, single output category classification~~ (Done)
  * ~~Multi parameter, single output category classification~~ (Done)
  * ~~Multi parameter, multi output category classiciation~~ (Done)
    * ~~One v. All Implementation~~ (Done)
* Neural Networks (TDB)
  * Architecture Model Flexibility (TBD)
  * Back Propagation (TBD)
  * Forward Propagation (TBD)
* Support Vector Machines (TBD)
* Optimizations
  * OPENCL Acceleration
    * ~~Vectorize Matrix arithmetic~~ (Done)
    * ~~Use OPENCL to accelerate matrix arithmeic~~ (Done)
    * Vectorize and use OPENCL to accelerate gradient descent for regression/classfication (TBD)


## Linear Algebra Library
## Regression Library
## Classification Library
## Neural Networks Library
