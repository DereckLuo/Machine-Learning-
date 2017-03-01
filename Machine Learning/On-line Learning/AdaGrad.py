# Class AdaGrad
# construct the AdaGrad model
# able to train from input numpy array and test on given test data set

import numpy as np
import math

# class AdaGrad model
class AdaGrad:
    # AdaGrade model parameters
    w = np.empty(1)     # weight vector in numpy array
    theta = 0.0         # something random
    eta = 1.0           # learning rate
    Gtjx = []           # the gradient square sum
    Gtjy = 0.0

    # given input data type, initialize the AdaGrad model parameters
    def __init__(self, x_size, eta):
        self.w = np.zeros(x_size)
        self.theta = 0.0
        self.eta = eta
        self.Gtjx = [0.0]*(x_size)
        self.Gtjy = 0.0

    # reset the model for new training
    def reset(self, eta):
        self.w = np.zeros(len(self.w))
        self.theta = 0.0
        self.eta = eta
        self.Gtjx = [0.0] * (len(self.w))
        self.Gtjy = 0.0

    # given input training data x and y, train AdaGrad parameters
    def train(self, x, y):
        mistake = 0
        mistake_arr = []
        for num in range(len(x)):
            yval = y[num]
            temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            if temp <= 0:
                mistake += 1
            # case need to update
            if temp <= 1:
                # calculate new gradient and update
                # temp1 = np.multiply(-yval, x[num])
                # temp2 = np.power(temp1, 2)
                self.Gtjx = self.Gtjx + np.power(np.multiply(-yval, x[num]), 2)
                root = np.sqrt(self.Gtjx)
                for i in range(len(self.w)):
                    # new_grad = -yval * x[num][i]
                    # self.Gtj[i] = self.Gtj[i] + new_grad**2
                    # root = math.sqrt(self.Gtjx[i])
                    if root[i] != 0.0:             # if root is 0, do not update
                        self.w[i] = self.w[i] + (self.eta*yval*x[num][i]) / root[i]
                # self.Gtj[len(self.w)] = self.Gtj[len(self.w)] + (-yval)**2
                self.Gtjy = self.Gtjy + (-yval)**2
            mistake_arr.append(mistake)
        return mistake_arr
        #print("Number of mistakes during AdaGrad learning is : ", mistake)

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
                if len(mistake_arr) > 10000:
                    return mistake_arr, mistake
                if mistake > 1700:
                    return mistake_arr, mistake
                yval = y[num]
                temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
                if temp <= 0:
                    mistake += 1
                    conti = 0
                else:
                    conti += 1
                # case need to update
                if temp <= 1:
                    # calculate new gradient and update
                    self.Gtjx = self.Gtjx + np.power(np.multiply(-yval, x[num]), 2)
                    root = np.sqrt(self.Gtjx)
                    for i in range(len(self.w)):
                        if root[i] != 0.0:  # if root is 0, do not update
                            self.w[i] = self.w[i] + (self.eta * yval * x[num][i]) / root[i]
                    # self.Gtj[len(self.w)] = self.Gtj[len(self.w)] + (-yval)**2
                    self.Gtjy = self.Gtjy + (-yval) ** 2
                mistake_arr.append(mistake)
                if conti >= d:
                    break
        return mistake_arr, mistake
        #print("Number of mistakes during AdaGrad learning is : ", mistake)

    # training function, which trains the algorithm and returns mistake and lose value
    def trainLose(self, x, y):
        mistake = 0
        lose = 0
        for num in range(len(x)):
            yval = y[num]
            temp = yval * (np.dot(np.transpose(self.w), x[num]) + self.theta)
            lose += max(0, 1-temp)
            if temp <= 0:
                mistake += 1
            # case need to update
            if temp <= 1:
                # calculate new gradient and update
                # temp1 = np.multiply(-yval, x[num])
                # temp2 = np.power(temp1, 2)
                self.Gtjx = self.Gtjx + np.power(np.multiply(-yval, x[num]), 2)
                root = np.sqrt(self.Gtjx)
                for i in range(len(self.w)):
                    # new_grad = -yval * x[num][i]
                    # self.Gtj[i] = self.Gtj[i] + new_grad**2
                    # root = math.sqrt(self.Gtjx[i])
                    if root[i] != 0.0:             # if root is 0, do not update
                        self.w[i] = self.w[i] + (self.eta*yval*x[num][i]) / root[i]
                # self.Gtj[len(self.w)] = self.Gtj[len(self.w)] + (-yval)**2
                self.Gtjy = self.Gtjy + (-yval)**2
        return mistake, lose

    # given input testing data x and y, test AdaGrad model and output prediction rate
    def test(self, x, y):
        correct = 0
        total = len(x)
        for num in range(len(x)):
            val = np.dot(np.transpose(self.w), x[num]) + self.theta
            if val > 0:
                predict = 1
            else:
                predict = -1
            if predict == y[num]:
                correct += 1
        return correct / total
