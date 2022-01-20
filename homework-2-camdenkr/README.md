# Homework 2
## Please Note For Solutions File:

1) Problem 1c was asked to plot as the original figure shows. Since the orginal figure is mostly about the decision boundaries and can't possibly also be showing the classification results, the points are colored for logisitic regression classification only, although since the end result in my plots are the same, it matters not.

2) "dataset" was used to consistently refer to the complete np array representation read from the provided MNIST files. The following variables are not arbitrary and were used consistently to refer to the same things as they are common variables used to refer the type of data they represent: "X" - mnist dataset matrix/array, "y" - labels for the mnist dataset, "x1, x2, ...xn" - nth dimension of the dataset (Note: this is not the case in Problem 1 where xn, yn are used since different sets of points had to be generated and they are in 2 dimensions, where x1 refers to the 1st set of points' x values, and y1 is the first set of points y values)

3) Logistic Regression does not converge in problem 2b, considering the accuracy and runtime with the default number of maximum iterations (and the excessive runtime of higher number of maximum iterations which still do not converges and reduce accuracy), I decided this was not an issue and allowed it to not converge.

4) Many of the input/outputs of various functions were modified, the functions' documentation should note them all. Too many were done at different times to note them all here.
