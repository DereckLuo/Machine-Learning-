# Class tuning
# perform tuning for all models on the given data
# tuning strategy: generate two distinct subsample of the training data
#   each consisting of 10% of the data set (D1, D2)
#   train algorithm on D1 by running algorithm 20 tims over the data
#   evaluate on D2, and record result

import numpy as np
import perceptron
import winnow
import AdaGrad

# class tuning
class tuning:
    datax = np.empty(1)
    datay = np.empty(1)
    pt_eta = [1.5, 0.25, 0.03, 0.005, 0.001]
    win_alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    win_gamma = [2.0, 0.3, 0.04, 0.005, 0.001]
    Ada_eta = [1.5, 0.25, 0.03, 0.005, 0.001]
    l = 0
    m = 0
    n = 0

    # load new training data for model
    def load(self, x, y, l, m, n):
        self.datax = x
        self.datay = y
        self.l = l
        self.m = m
        self.n = n

    # perceptron tuning
    # perform tuning on perceptron(with margin)\
    # perceptron without margin do not require tuning
    # eta : array of parameters need to be tried
    def tunPerceptron(self, model, eta, D1x, D1y, D2x, D2y):
        print("Tunning Perceptron(with margin) model starts ...")
        best_eta = -1
        best_result = 0
        # split data into D1 and D2
        # D1x, D1y, D2x, D2y = self.datasplit()

        for i in eta:
            model.reset(i)  # reset model
            for j in range(20):
                model.train(D1x, D1y)
            result = model.test(D2x, D2y)
            print(" Parameter eta = ", i, " accuracy is : ", result)
            if result > best_result:
                best_result = result
                best_eta = i
        return best_eta, best_result

    # winnow tuning
    # perform tuning on winnow(without margin)
    # alpha: array of parameters need to be tried
    def tunWinnowNoMargin(self, model, alpha, D1x, D1y, D2x, D2y):
        print("Tuning Winnow(without margin) model starts ...")
        best_alpha = -1
        best_result = 0
        # split data into D1 and D2
        # D1x, D1y, D2x, D2y = self.datasplit()

        for i in alpha:
            model.reset(i)
            for j in range(20):
                model.train(D1x, D1y)
            result = model.test(D2x, D2y)
            print(" Parameter alpha = ", i, " accuracy is : ", result)
            if result > best_result:
                best_result = result
                best_alpha = i
        return best_alpha, best_result

    # winnow tuning
    # perform tuning on winnow(with margin)
    # alpha: array of parameters need to be tried
    # gamma: array of parameters need to be tried
    def tunWinnowMargin(self, model, alpha, gamma, D1x, D1y, D2x, D2y):
        print("Tunning Winnow(with margin) model starts ...")
        best_alpha = -1
        best_gamma = -1
        best_result = 0
        # D1x, D1y, D2x, D2y = self.datasplit()

        for i in alpha:
            for j in gamma:
                model.reset(i, j)
                for k in range(20):
                    model.train(D1x, D1y)
                result = model.test(D2x, D2y)
                print(" Parameter alpha = ", i, " Parameter gamma = ", j, " accuracy is : ", result)
                if result > best_result:
                    best_alpha = i
                    best_gamma = j
                    best_result = result
        return best_alpha, best_gamma, best_result

    # AdaGrad tuning
    # perform tuning on AdaGrad
    # eta : array of parameters ned to be tried
    def tunAdaGrad(self, model, eta, D1x, D1y, D2x, D2y):
        print("Tuning AdaGrad model starts ...")
        best_eta = -1
        best_result = 0
        # D1x, D1y, D2x, D2y = self.datasplit()

        for i in eta:
            model.reset(i)
            for j in range(20):
                model.train(D1x, D1y)
            result = model.test(D2x, D2y)
            print(" Parameter eta = ", i, " accuracy is : ", result)
            if result > best_result:
                best_eta = i
                best_result = result
        return best_eta, best_result

    # datasplit
    # function to randomly spit data into two 10% subset
    # D1 for training, D2 for testing
    def datasplit(self):
        length = len(self.datax)
        tenP = round(length / 10)
        indices = np.random.permutation(self.datax.shape[0])
        D1idx, D2idx = indices[:tenP], indices[length - tenP:]
        D1x = np.take(self.datax, D1idx, axis=0)
        D1y = np.take(self.datay, D1idx, axis=0)
        D2x = np.take(self.datax, D2idx, axis=0)
        D2y = np.take(self.datay, D2idx, axis=0)
        return D1x, D1y, D2x, D2y

    # allmodelTun
    # function to perform tunning on all models with given dataset
    def allmodelTun(self):
        # generate tunning data
        D1x, D1y, D2x, D2y = self.datasplit()

        # tuning perceptron with margin
        pt = perceptron.perceptron_margin(len(self.datax[0]), 0)
        pteta, ptresult = self.tunPerceptron(pt, self.pt_eta, D1x, D1y, D2x, D2y)
        print("Data l, m, n : ", self.l, self.m, self.n, "~~~~~~~~~~~ Best Perceptron eta is : ", pteta, " result is : ", ptresult)

        # tuning winnow with no margin
        winnowNoM = winnow.winnow_nomargin(len(self.datax[0]), 0)
        winNalpha, winNresult = self.tunWinnowNoMargin(winnowNoM, self.win_alpha, D1x, D1y, D2x, D2y)
        print("Data l, m, n : ", self.l, self.m, self.n, "~~~~~~~~~~~ Best winN alpha1 is : ", winNalpha, " result is : ", winNresult)

        # tuning winnow with margin
        winnowM = winnow.winnow_margin(len(self.datax[0]),0,0)
        winMalpha, winMgamma, winMresult = self.tunWinnowMargin(winnowM, self.win_alpha, self.win_gamma, D1x, D1y, D2x, D2y)
        print("Data l, m, n : ", self.l, self.m, self.n, "~~~~~~~~~~~ Best winM alpha1 is : ", winMalpha, " Best winM gamma1 is : ", winMgamma, " result is : ", winMresult)

        # tuning AdaGrad
        Ada = AdaGrad.AdaGrad(len(self.datax[0]), 0)
        Adaeta, Adaresult = self.tunAdaGrad(Ada, self.Ada_eta, D1x, D1y, D2x, D2y)
        print("Data l, m, n : ", self.l, self.m, self.n, "~~~~~~~~~~~ Best Ada eta1 is : ", Adaeta, " result is : ", Adaresult)

        return pteta, winNalpha, winMalpha, winMgamma, Adaeta

