####    CS 446 Machine Learning Homework 3
####    Author : Chongxin Luo
####    2/25/2017

import numpy as np
import testing
import gen
import tuning
import perceptron
import winnow
import AdaGrad
import pylab
from matplotlib import pyplot as plt


# # merge data
# # function concatenate x and y dataset into a single dataset with y concatenated on the end of x
# def mergeData(x, y):
#     y.resize(len(y), 1)
#     dataset = np.append(x, y, axis=1)
#     return dataset


# Function Question1
# running question and output results of question 1
def Question1():
    # (a). Generating two dataset
    dataset1y, dataset1x = gen.gen(10, 100, 500, 50000, False)
    dataset2y, dataset2x = gen.gen(10, 100, 5000, 50000, False)

    # (b). Tuning parameters, record optimal parameters
    tune1 = tuning.tuning()
    tune2 = tuning.tuning()
    tune1.load(dataset1x, dataset1y, 10, 100, 500)
    tune2.load(dataset2x, dataset2y, 10, 100, 5000)

    pteta1, winNalpha1, winMalpha1, winMgamma1, Adaeta1 = tune1.allmodelTun()
    pteta2, winNalpha2, winMalpha2, winMgamma2, Adaeta2 = tune2.allmodelTun()
    print(pteta1, winNalpha1, winMalpha1, winMgamma1, Adaeta1)
    print(pteta2, winNalpha2, winMalpha2, winMgamma2, Adaeta2)

    # (c,d) running best parameter on entire training set. Plot mistake vs sample n
    #  --- training set with n = 500 ---
    # trainMistakePlot(dataset1x, dataset1y, 0.005, 1.1, 1.1, 0.04, 0.25)
    trainMistakePlot(dataset1x, dataset1y, pteta1, winNalpha1, winMalpha1, winMgamma1, Adaeta1)
    #  --- training set with n = 5000 ---
    # trainMistakePlot(dataset2x, dataset2y, 0.001, 1.01, 1.1, 0.04, 0.03)
    trainMistakePlot(dataset2x, dataset2y, pteta2, winNalpha2, winMalpha2, winMgamma2, Adaeta2)

# function to train and using converge
def trainConvergePlot(datasetx, datasety, d, pteta, winNalpha, winMalpha, winMgamma, Adaeta):
    # perceptron with no margin
    print("Perceptron with no margin full training ...")
    ptN = perceptron.perceptron_nomargin(len(datasetx[0]))
    ptN_mistakearr, ptN_mistake = ptN.trainConverge(datasetx, datasety, d)
    print(ptN_mistake)

    #perceptron with margin
    print("Perceptron with margin full training ...")
    ptM = perceptron.perceptron_margin(len(datasetx[0]), pteta)
    ptM_mistakearr, ptM_mistake = ptM.trainConverge(datasetx, datasety, d)
    print(ptM_mistake)

    # Winnow with no margin
    print("Winnow with no margin full training ...")
    winN = winnow.winnow_nomargin(len(datasetx[0]), winNalpha)
    winN_mistakearr, winN_mistake = winN.trainConverge(datasetx, datasety, d)
    print(winN_mistake)

    # Winnow with margin
    print("Winnow with margin full training ...")
    winM = winnow.winnow_margin(len(datasetx[0]), winMalpha, winMgamma)
    winM_mistakearr, winM_mistake = winM.trainConverge(datasetx, datasety, d)
    print(winM_mistake)

    # AdaGrad model
    print("AdaGrad full training ...")
    Ada = AdaGrad.AdaGrad(len(datasetx[0]), Adaeta)
    Ada_mistakearr, Ada_mistake = Ada.trainConverge(datasetx, datasety, 10)
    print(Ada_mistake)

    # plotting mistake vs sample n on converge
    return ptN_mistake, ptN_mistakearr, ptM_mistake, ptM_mistakearr, winN_mistake, winN_mistakearr, winM_mistake, winM_mistakearr, Ada_mistake, Ada_mistakearr



# function to train and plot out mistake vs sample n for all models
def trainMistakePlot(datasetx, datasety, pteta, winNalpha, winMalpha, winMgamma, Adaeta):
    # perceptron with no margin
    print("Perceptron with no margin full training ...")
    ptN = perceptron.perceptron_nomargin(len(datasetx[0]))
    ptN_mistake = ptN.train(datasetx, datasety)
    print(ptN_mistake[len(ptN_mistake)-1])
    # perceptron with margin
    print("Perceptron with margin full training ...")
    ptM = perceptron.perceptron_margin(len(datasetx[0]), pteta)
    ptM_mistake = ptM.train(datasetx, datasety)
    print(ptM_mistake[len(ptM_mistake)-1])

    # Winnow with no margin
    print("Winnow with no margin full training ...")
    winN = winnow.winnow_nomargin(len(datasetx[0]), winNalpha)
    winN_mistake = winN.train(datasetx, datasety)
    print(winN_mistake[len(winN_mistake)-1])
    # Winnow with margin
    print("Winnow with margin full training ...")
    winM = winnow.winnow_margin(len(datasetx[0]), winMalpha, winMgamma)
    winM_mistake = winM.train(datasetx, datasety)
    print(winM_mistake[len(winM_mistake)-1])

    # AdaGrad model
    print("AdaGrad full training ...")
    Ada = AdaGrad.AdaGrad(len(datasetx[0]), Adaeta)
    Ada_mistake = Ada.train(datasetx, datasety)
    print(Ada_mistake[len(Ada_mistake)-1])

    # plotting mistake vs sample n
    pylab.plot(ptN_mistake, 'r', label='perceptron No margin')
    pylab.plot(ptM_mistake, 'g', label = 'perceptron with margin')
    pylab.plot(winN_mistake, 'b', label='Winnows No margin')
    pylab.plot(winM_mistake, 'c', label='Winnows with margin')
    pylab.plot(Ada_mistake, 'y', label = 'AdaGrad model')
    pylab.legend(loc='upper left')
    pylab.xlabel('number of examples N')
    pylab.ylabel('number of mistakes M')
    pylab.show()
    # plotname = input('input plot name : ')
    # pylab.savefig(plotname)

# Function Question2
# running question and output results of question 2
def Question2():
    l = 10
    m = 20
    ptNmistake = []
    ptNlocation = []
    ptMmistake = []
    ptMlocation = []
    winNmistake = []
    winNlocation = []
    winMmistake = []
    winMlocation = []
    Adamistake = []
    Adalocation = []
    for n in range(40, 240, 40):
        datasety, datasetx = gen.gen(l, m, n, 50000, False)
        tune = tuning.tuning()
        tune.load(datasetx, datasety, l, m, n)
        pteta, winNalpha, winMalpha, winMgamma, Adaeta = tune.allmodelTun()
        ptN_mistake, ptN_mistakearr, ptM_mistake, ptM_mistakearr, winN_mistake, winN_mistakearr, winM_mistake, winM_mistakearr, Ada_mistake, Ada_mistakearr = trainConvergePlot(datasetx, datasety, 1000, pteta, winNalpha, winMalpha, winMgamma, Adaeta)
        ptNmistake.append(ptN_mistake)
        ptNlocation.append(ptN_mistakearr[len(ptN_mistakearr)-1])
        ptMmistake.append(ptM_mistake)
        ptMlocation.append(ptM_mistakearr[len(ptM_mistakearr)-1])
        winNmistake.append(winN_mistake)
        winNlocation.append(winN_mistakearr[len(winN_mistakearr)-1])
        winMmistake.append(winM_mistake)
        winMlocation.append(winM_mistakearr[len(winM_mistakearr)-1])
        Adamistake.append(Ada_mistake)
        Adalocation.append(Ada_mistakearr[len(Ada_mistakearr)-1])

        fig = plt.figure()
        plt.plot(ptN_mistakearr, 'r', label = 'perceptron No margin')
        plt.plot(ptM_mistakearr, 'g', label = 'perceptron with margin')
        plt.plot(winN_mistakearr, 'b', label = 'Winnows No margin')
        plt.plot(winM_mistakearr, 'c', label = 'Winnows with margin')
        plt.plot(Ada_mistakearr, 'y', label = 'AdaGrad model')
        plt.legend(loc='lower right')
        pylab.xlabel('number of examples N')
        pylab.ylabel('number of mistakes M')

        plt.show()

    # plotting mistake vs sample n at converge time

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(ptNlocation, ptNmistake, 'r*', label = 'perceptron No margin')
    plt.plot(ptMlocation, ptMmistake, 'g*', label = 'perceptron with margin')
    plt.plot(winNlocation, winNmistake, 'b*', label = 'Winnows No margin')
    plt.plot(winMlocation, winMmistake, 'c*', label = 'Winnows with margin')
    plt.plot(Adalocation, Adamistake, 'y*', label = 'AdaGrad model')
    plt.legend(loc='upper left')
    pylab.xlabel('number of examples N')
    pylab.ylabel('number of mistakes M')

    for i in range(len(Adalocation)):
        ax.annotate('n=%s,#mistake=%s' %(ptNlocation[i], ptNmistake[i]), xy=(ptNlocation[i], ptNmistake[i]), textcoords = 'data')
        ax.annotate('n=%s,#mistake=%s' % (ptMlocation[i], ptMmistake[i]), xy=(ptMlocation[i], ptMmistake[i]), textcoords='data')
        ax.annotate('n=%s,#mistake=%s' % (winNlocation[i], winNmistake[i]), xy=(winNlocation[i], winNmistake[i]), textcoords='data')
        ax.annotate('n=%s,#mistake=%s' % (winMlocation[i], winMmistake[i]), xy=(winMlocation[i], winMmistake[i]), textcoords='data')
        ax.annotate('n=%s,#mistake=%s' % (Adalocation[i], Adamistake[i]), xy=(Adalocation[i], Adamistake[i]), textcoords='data')

    plt.show()

# function Question3
# running question and output result of question 3
def Question3():
    l = 10
    n = 1000
    m = [100, 500, 1000]

    for i in range(2,3):
        # (a) Data Generation
        (trainy, trainx) = gen.gen(l,m[i],n,50000,True)
        (testy, testx) = gen.gen(l,m[i],n,10000,False)

        # (b) Parameter Tune
        tune = tuning.tuning()
        tune.load(trainx, trainy, l, m[i], n)
        pteta, winNalpha, winMalpha, winMgamma, Adaeta = tune.allmodelTun()

        # (c) Training
        ptN = perceptron.perceptron_nomargin(len(trainx[0]))
        ptM = perceptron.perceptron_margin(len(trainx[0]), pteta)
        winN = winnow.winnow_nomargin(len(trainx[0]), winNalpha)
        winM = winnow.winnow_margin(len(trainx[0]), winMalpha, winMgamma)
        Ada = AdaGrad.AdaGrad(len(trainx[0]), Adaeta)
        for j in range(20):
            ptN.train(trainx, trainy)
            ptM.train(trainx, trainy)
            winN.train(trainx, trainy)
            winM.train(trainx, trainy)
            Ada.train(trainx, trainy)

        # (d) Testing
        ptNresult = ptN.test(testx, testy)
        ptMresult = ptM.test(testx, testy)
        winNresult = winN.test(testx, testy)
        winMresult = winM.test(testx, testy)
        Adaresult = Ada.test(testx, testy)

        print(pteta, winNalpha, winMalpha, winMgamma, Adaeta)
        print(ptNresult, ptMresult, winNresult, winMresult, Adaresult)


# function Bonus
# running question bonus and output result of question
def Bonus():
    l = 10
    m = 20
    n = 80

    for n in range(40, 200, 40):
        data_y, data_x = gen.gen(l, m, n, 10000, True);

        Ada=AdaGrad.AdaGrad(len(data_x[0]), 0.25)

        mistake_arr = []
        lose_arr = []

        for i in range(50):
            mistake, lose = Ada.trainLose(data_x, data_y)
            mistake_arr.append(mistake)
            lose_arr.append(lose)

        plt.plot(mistake_arr, 'r', label = 'Misclassification error')
        plt.xlabel('number of training sessions N')
        plt.ylabel('error value')
        plt.show()

        plt.plot(lose_arr, 'g', label='Hinge lose')
        plt.xlabel('number of training sessions N')
        plt.ylabel('error value')
        plt.show()


if __name__ == "__main__":
    while True:
        print(" ----- Homework experiment running ---- ")
        print("part1 : Number of examples versus number of mistakes")
        print("part2 : Learning curves of online learning algorithms")
        print("part3 : Use online learning algorithms as batch learning algorithms")
        print("part4 : Bonus")
        run = input("Please enter input experiments : ")

        # case running experiment 1
        if run == 'part1':
            print("Running experiment 1 ...")
            Question1()

        # case running experiment 2
        elif run == 'part2':
            print("Running experiment 2 ...")
            Question2()

        # case running experiment 3
        elif run == 'part3':
            print("Running experiment 3 ...")
            Question3()

        # case running experiment 4
        elif run == 'part4':
            print("Running bonus experiment")
            Bonus()

        # exist program
        elif run == 'exit':
            break

        # default statement
        else:
            print("Wrong experiment number input, please try again")
