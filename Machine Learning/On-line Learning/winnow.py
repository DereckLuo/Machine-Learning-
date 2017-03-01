# Class winnow
# contrain winnow model
# able to train from input numpy array and test on given test data set

import numpy as np


# class winnow model without margin
class winnow_nomargin:
    # winnow (without margin) parameters
    w = np.empty(1)  # weight vector
    theta = -1.0  # set as -n, do not change
    alpha = 0  # promotion parameter

    # given input data type, initial parameter value, init winnow model
    def __init__(self, x_size, alpha):
        self.w = np.ones(x_size)
        self.theta = -1.0
        self.alpha = alpha

    # reset the model for new training
    def reset(self, alpha):
        self.w = np.ones(len(self.w))
        self.theta = -1.0
        self.alpha = alpha

    # given input training data x and y, train winnow parameters
    def train(self, x, y):
        mistake = 0
        mistake_arr = []
        for num in range(len(x)):
            yval = y[num]
            temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            # compute mistake
            if temp <= 0:
                mistake += 1
                self.w = np.multiply(self.w, (np.power(self.alpha, (np.multiply(yval, x[num])))))
                # for i in range(len(x[num])):
                #     self.w[i] = self.w[i] * (self.alpha ** (yval*x[num][i]))
            mistake_arr.append(mistake)
        return mistake_arr
        #print("Number of mistakes during Winnow(without margin) learning is : ", mistake)

    # train model using converge method
    def trainConverge(self, x, y, d):
        mistake = 0
        mistake_arr = []
        conti = 0
        iteration = 0
        while(conti < d):
            iteration += 1
            print(" on iteration : ", iteration)
            for num in range(len(x)):
                yval = y[num]
                temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
                # compute mistake
                if temp <= 0:
                    mistake += 1
                    conti = 0
                    self.w = np.multiply(self.w, (np.power(self.alpha, (np.multiply(yval, x[num])))))
                else:
                    conti += 1
                mistake_arr.append(mistake)
                if conti >= d:
                    break
        return mistake_arr, mistake

    # given input testing data x and y, test winnow model and output prediction rate
    def test(self, x, y):
        return winnowTest(x, y, self.w, self.theta)


# class winnow model with margin
class winnow_margin:
    # winnow (with margin) parameters
    w = np.empty(1)  # weight vector
    theta = -1.0  # set as -n, do not change
    alpha = 0.0  # promotion parameter
    gamma = 1  # margin threshold

    # given input data type and initial winnow model
    def __init__(self, x_size, alpha, gamma):
        self.w = np.ones(x_size)
        self.theta = -1.0
        self.alpha = alpha
        self.gamma = gamma

    # reset the model for new training
    def reset(self, alpha, gamma):
        self.w = np.ones(len(self.w))
        self.theta = -1.0
        self.alpha = alpha
        self.gamma = gamma

    # given input training data x and y, train winnow model
    def train(self, x, y):
        mistake = 0
        mistake_arr = []
        for num in range(len(x)):
            yval = y[num]
            # compute y(wTx + theta)
            temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            # count the mistakes
            if temp <= 0:
                mistake += 1
            # case update the parameter
            if temp <= self.gamma:
                self.w = np.multiply(self.w, (np.power(self.alpha, (np.multiply(yval, x[num])))))
                # for i in range(len(x[num])):
                #     self.w[i] = self.w[i] * (self.alpha ** (yval * x[num][i]))
            mistake_arr.append(mistake)
        return mistake_arr
        #print("Number of mistakes during Winnow(with margin) learning is : ", mistake)

        # train model using converge method
    def trainConverge(self, x, y, d):
        mistake = 0
        mistake_arr = []
        conti = 0
        iteration = 0
        while (conti < d):
            iteration += 1
            print(" on iteration : ", iteration)
            for num in range(len(x)):
                yval = y[num]
                temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
                # compute mistake
                if temp <= 0:
                    mistake += 1
                    conti = 0
                else:
                    conti += 1
                if temp < self.gamma:
                    self.w = np.multiply(self.w, (np.power(self.alpha, (np.multiply(yval, x[num])))))
                mistake_arr.append(mistake)
                if conti >= d:
                    break
        return mistake_arr, mistake

    # given input testing data x and y, test winnow moddel and output prediction rate
    def test(self, x, y):
        return winnowTest(x, y, self.w, self.theta)


# General Winnow test function
# given test data x and y, model w and theta
# return test accuracy
def winnowTest(x, y, w, theta):
    correct = 0
    total = len(x)
    for num in range(len(x)):
        val = np.dot(np.transpose(w), x[num]) + theta
        if (val > 0):
            predict = 1
        else:
            predict = -1
        if predict == y[num]:
            correct += 1
    return correct / total
