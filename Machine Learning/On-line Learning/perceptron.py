# Class perceptron
# contrain perceptron model
# able to train from input numpy array and test on given test data set

import numpy as np

# class perceptron model without margin implementation
class perceptron_nomargin:
    # perceptron (without margin) parameters
    w = np.empty(1) # weight vector in numpy array
    theta = 0.0      # something random lol
    eta = 1.0        # learning rate


    # given input data type, initialize the perception parameters
    def __init__(self, x_size):
        self.w = np.zeros(x_size)
        self.theta = 0.0
        self.eta = 1.0

    # reset the model for new training
    def reset(self):
        self.w = np.zeros(len(self.w))
        self.theta = 0
        self.eta = 1.0

    # given input training data x and y, train perceptron parameters
    def train(self, x, y):
        mistake = 0
        mistake_arr = []
        for num in range(len(x)):
            # compute y(wTx + theta)
            temp = y[num] * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            # case update the parameters
            if temp <= 0:
                mistake += 1
                self.w = self.w + np.dot(self.eta * y[num], x[num])
                self.theta = self.theta + self.eta * y[num]
            mistake_arr.append(mistake)
        return mistake_arr
        #print("Number of mistakes during Perceptron(without margin) learning is : " , mistake)

    # training model using converge method
    def trainConverge(self, x, y, d):
        mistake = 0
        mistake_arr = []
        conti = 0
        iteration = 0
        while(conti < d):
            iteration += 1
            print(" on iteration : ", iteration)
            for num in range(len(x)):
                # compute y(wTx + theta)
                temp = y[num] * (np.dot(np.transpose(self.w), x[num]) + self.theta)
                # case update the parameters
                if temp <= 0:
                    mistake += 1
                    conti = 0
                    self.w = self.w + np.dot(self.eta * y[num], x[num])
                    self.theta = self.theta + self.eta * y[num]
                else:
                    conti += 1
                mistake_arr.append(mistake)
                if conti >= d:
                    break
        return mistake_arr, mistake


    # given input testing data x and y, test perceptron and output error rate
    def test(self, x, y):
        return perceptronTest(x,y,self.w, self.theta)



# class perceptron model with margin implementation
class perceptron_margin:
    # perceptron (with margin) parameters
    w = np.empty(1)
    theta = 0
    eta = 0
    gamma = 1.0

    # given input data type, and initial parameters, init class
    def __init__(self, x_size, eta):
        self.w = np.zeros(x_size)   # weight vector
        self.theta = 0.0  # something random lol
        self.eta = eta  # learning rate
        self.gamma = 1.0  # margin threshold

    # reset model to new parameters for new training
    def reset(self, eta):
        self.w = np.zeros(len(self.w))
        self.theta = 0.0
        self.eta = eta
        self.gamma = 1.0

    # given input training data x and y, train perceptron
    def train(self, x, y):
        mistake = 0
        mistake_arr = []
        for num in range(len(x)):
            # compute y(wTx + theta)
            temp = y[num] * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            # count the mistake
            if temp <= 0:
                mistake += 1
            # case update the parameters
            if temp <= self.gamma:
                self.w = self.w + np.dot(self.eta * y[num], x[num])
                self.theta = self.theta + self.eta * y[num]
            mistake_arr.append(mistake)
        return mistake_arr
        #print("Number of mistakes during Perceptron(with margin) learning is : " , mistake)

    # training model using converge method
    def trainConverge(self, x, y, d):
        mistake = 0
        mistake_arr = []
        conti = 0
        iteration = 0
        while (conti < d):
            iteration += 1
            print(" on iteration : ", iteration)
            for num in range(len(x)):
                # compute y(wTx + theta)
                temp = y[num] * (np.dot(np.transpose(self.w), x[num]) + self.theta)
                # case update the parameters
                if temp <= 0:
                    mistake += 1
                    conti = 0
                else:
                    conti += 1
                if temp <= self.gamma:
                    self.w = self.w + np.dot(self.eta * y[num], x[num])
                    self.theta = self.theta + self.eta * y[num]
                mistake_arr.append(mistake)
                if conti >= d:
                    break
        return mistake_arr, mistake

    # given input testing data x and y, test perceptron and output error rate
    def test(self, x, y):
        return perceptronTest(x, y, self.w, self.theta)

# General perceptron test function
# give test data x and y, model w and theta
# return test accuracy
def perceptronTest(x, y, w, theta):
    correct = 0
    total = len(x)
    for num in range(len(x)):
        val = np.dot(np.transpose(w), x[num]) + theta
        if(val > 0):
            predict = 1
        else:
            predict = -1
        if predict == y[num]:
            correct += 1
    return correct/total
