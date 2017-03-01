# Testing file
# contains all test functions for all models

import add_noise
import gen
import perceptron
import winnow
import AdaGrad
import tuning

# perceptron test
def perceptronTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    dataset2y, dataset2x = gen.gen(10, 100, 1000, 50000, False)
    pNoMargin = perceptron.perceptron_nomargin(len(dataset1x[0]))
    pNoMargin.train(dataset1x, dataset1y)
    output = pNoMargin.test(dataset1x, dataset1y)
    print(output)
    pMargin = perceptron.perceptron_margin(len(dataset2x[0]), 1.5)
    pMargin.reset(1.5)
    pMargin.train(dataset2x, dataset2y)
    output = pMargin.test(dataset2x, dataset2y)
    print(output)

# Winnow test
def winnowTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    wNoMargin = winnow.winnow_nomargin(len(dataset1x[0]), 1.005)
    wNoMargin.train(dataset1x, dataset1y)
    output = wNoMargin.test(dataset1x, dataset1y)
    print(output)
    wMargin = winnow.winnow_margin(len(dataset1x[0]), 1.01, 0.04)
    wMargin.train(dataset1x, dataset1y)
    output = wMargin.test(dataset1x, dataset1y)
    print(output)

# AdaGrad test
def AdaGradTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    Ada = AdaGrad.AdaGrad(len(dataset1x[0]), 1.5)
    Ada.train(dataset1x, dataset1y)
    output = Ada.test(dataset1x, dataset1y)
    print(output)

# Perceptron Tuning Test
def perceptronTuneTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    tune = tuning.tuning()
    pt = perceptron.perceptron_margin(len(dataset1x[0]), 0)
    eta = [1.5, 0.25, 0.03, 0.005, 0.001]

    tune.load(dataset1x, dataset1y)
    best_eta, best_result = tune.tunPerceptron(pt, eta)
    print("Best eta is : ", best_eta)
    print("Best result is : ", best_result)

# Winnow tunning test
def winnowTunTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    tune = tuning.tuning()
    tune.load(dataset1x, dataset1y)

    winnowNoM = winnow.winnow_nomargin(len(dataset1x[0]), 0)
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    best_alpha, best_result = tune.tunWinnowNoMargin(winnowNoM, alpha)
    print("No margin best alpha is : ", best_alpha)
    print("No margin best result is : ", best_result)

    winnowM = winnow.winnow_margin(len(dataset1x[0]), 0, 0)
    gamma = [2.0, 0.3, 0.04, 0.006, 0.001]
    best_alpha, best_gamma, best_result = tune.tunWinnowMargin(winnowM, alpha, gamma)
    print("With margin best alpha is : ", best_alpha)
    print("With margin best gamma is : ", best_gamma)
    print("With margin best result is : ", best_result)

# AdaGrad tuning test
def AdaGradTunTest():
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    tune = tuning.tuning()
    tune.load(dataset1x, dataset1y)

    Ada = AdaGrad.AdaGrad(len(dataset1x[0]), 0)
    eta = [1.5, 0.25, 0.03, 0.005, 0.001]
    best_eta, best_result = tune.tunAdaGrad(Ada, eta)
    print(" AdaGrad best eta is : " , best_eta)
    print(" AdaGrad best result is : ", best_result)