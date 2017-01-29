# a complete code for data preprocessing
from sklearn import datasets
from sklearn.cross_validation import train_test_split

import numpy as np
import pandas as pd
import random as rand

# dictionary format for dataSet RobotExecution
dict_1 = {'normal': 0, 'collision': 1, 'obstruction': 2, 'fr_collision': 3}
dict_2 = {'normal': 0, 'front_col':1, 'back_col':2, 'left_col':3, 'right_col':4}
dict_4 = {'normal':0, 'collision':1 ,'obstruction':2}
dict_iono = {'g':0, 'b':1}

# three map functions used on target of dataSet RobotExecution
def robotTargetMap_1(x):
    return dict_1[x]

def robotTargetMap_2(x):
    return dict_2[x]

def robotTargetMap_4(x):
    return dict_4[x]

def ionoMap(x):
    return dict_iono[x]

def selfAddByOne(x):
    """
    self add by one function
    :param x:
    :return:
    """
    return x + 1


def distEclud(vecA, vecB):
    """
    calculate the euclidean distance of 2 given vectors
    :param vecA:
    :param vecB:
    :return: euclidean distance
    """
    return np.sqrt(sum(np.power(vecA-vecB, 2)))


def RepetitionRandomSampling(dataSet, n_features):
    """
    random sampling by features (without replacement)
    :param dataSet: dataset to be sampled
    :param n_features: number of features
    :return: sampled dataSet (in a sub-space)
    """
    print ('RepetitionRandomSampling:')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    sample = []
    dataSet = dataSet.T
    for i in range(n_features):
        sample.append(dataSet[rand.randint(0, len(dataSet)-1)])
    #sample = np.array(sample)  # change list to np.array  with it or not is ok,for list can be directly converted into dataFrame type
    dataFrame = pd.DataFrame(sample)
    sample = dataFrame.drop_duplicates()
    sample = np.array(sample)
    return sample.T


def loadIris():
    """
    simply load the iris data
    :return: tuple(data, target)
    """
    print ('Load Iris:')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    iris = datasets.load_iris()
    return iris.data, iris.target


def loadDigits():
    """
        simply load the digits data
        :return: tuple(data, target)
        """
    print ('Load Digits:')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    digits = datasets.load_digits()
    return digits.data, digits.target


def loadFirstDataSet(fileName):
    """
    load the common dataSet (labels in the first column)
    :return: tuple(dataSet, target)
    """
    print ('Load DataSet')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(int(curLine[0]))
        curLine.remove(curLine[0])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    return np.array(dataSet), np.array(target)


def loadLastDataSet(fileName):
    """
    load the common dataSet (labels in the last column)
    :return: tuple(dataSet, target)
    """
    print ('Load DataSet')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(int(curLine[-1]))
        curLine = curLine[:-1]
        # curLine.remove(curLine[-1])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    return np.array(dataSet), np.array(target)


def loadMovement_libras():
    """
        load the 'Movement_libras' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Load Movement_libras')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet, target = loadLastDataSet('UCI Data/Movement_libras/movement_libras.data')
    target -= 1
    return dataSet, target


def loadSynthetic_control():
    """
        load the 'synthetic_control' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Load Synthetic_control')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    fr = open('UCI Data/Synthetic_control/synthetic_control.data')

    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    list = [0] * 100
    for i in range(6):
        target.extend(list)
        list = map(selfAddByOne, list)

    return np.array(dataSet), np.array(target)


def loadRobotExecution(fileName):
    """
        load the 'RobotExecution' dataSet
        :return: list(dataSet, target)
    """
    print ('Load RobotExecution')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    row = []
    fr = open(fileName)
    index = 1
    for line in fr.readlines():
        if not line.strip():
            continue
        curLine = line.strip().split()
        if index == 1:
            target.extend(curLine)
        else:
            intLine = map(int, curLine)
            row.extend(intLine)
        index += 1
        if index == 17:
            dataSet.append(row)
            index = 1
            row = []
    return dataSet, target


def loadRobotExecution_1():
    """
        load the 'lp1' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Load RobotExecution_1')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet, target = loadRobotExecution('UCI Data/RobotExecution/lp1.data')
    target = map(robotTargetMap_1, target)

    return np.array(dataSet), np.array(target)


def loadRobotExecution_2():
    """
        load the 'lp2' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Load RobotExecution_2')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet, target = loadRobotExecution('UCI Data/RobotExecution/lp2.data')
    target = map(robotTargetMap_2, target)
    return np.array(dataSet), np.array(target)


def loadRobotExecution_4():
    """
        load the 'lp4' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Load RobotExecution_4')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet, target = loadRobotExecution('UCI Data/RobotExecution/lp4.data')
    target = map(robotTargetMap_4, target)
    return np.array(dataSet), np.array(target)

def loadGlass():
    """
        load glass dataset
    :return:
    """
    print ('Load Glass...')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet, target = loadLastDataSet('UCI Data/glass/glass.data.txt')
    return dataSet[:, 1:], target - 1

def loadWine():
    """
    load wine dataset
    :return:
    """
    dataSet, target = loadFirstDataSet('UCI Data/wine/wine.data.txt')
    return dataSet, target - 1

def loadIonosphere():
    """
    load ionosphere
    :return:
    """
    print ('Load DataSet')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    fr = open('UCI Data/ionosphere/ionosphere.data.txt')

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(curLine[-1])
        curLine = curLine[:-1]
        # curLine.remove(curLine[-1])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    target = map(ionoMap, target)

    return np.array(dataSet), np.array(target)

def loadIsolet():
    """
            load isolet dataset
        :return:
        """
    print ('Load ISOLET...')
    print ('**********************************************************************************************************'
           '**********************************************************************************************************')
    dataSet = []
    target = []
    fr = open('UCI Data/ISOLET/isolet5.data')

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(int((curLine[-1]).replace('.', '')))
        curLine = curLine[:-1]
        # curLine.remove(curLine[-1])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    return np.array(dataSet), np.array(target) - 1


def load_gisette_data():
    aaa = np.loadtxt('/Users/Emrlay/gisette_train.data')
    labels = np.loadtxt('/Users/Emrlay/n_gisette_train.labels.txt')
    return aaa, labels


def Test():
    # Movement_libras test
    # dataSet, target = loadMovement_libras()
    # print (dataSet)
    # print(target)

    # Synthetic_control test
    # dataSet, target = loadSynthetic_control()

    # Digits test
    # dataSet, target = loadDigits()

    # RobotExecution test
    # dataSet, target = loadRobotExecution_4()

    # glass test
    # dataSet, target = loadGlass()

    # wine test
    # dataSet, target = loadWine()

    # ionosphere test
    # dataSet, target = loadIonosphere()

    dataSet, target = loadIsolet()
    print dataSet
    print target
    print (dataSet.shape)
    print (target.shape)



# Test()