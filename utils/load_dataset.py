"""
Load datasets from file
"""
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix

# dictionary format for dataSet RobotExecution
robot_label_mapper = {'normal': 0, 'collision': 1, 'obstruction': 2, 'fr_collision': 3}
robot2_label_mapper = {'normal': 0, 'front_col': 1, 'back_col': 2, 'left_col': 3, 'right_col': 4}
robot4_label_mapper = {'normal': 0, 'collision': 1, 'obstruction': 2}
segmentation_mapper = {'GRASS': 0 ,'PATH': 1, 'WINDOW': 2, 'CEMENT': 3, 'FOLIAGE': 4, 'SKY': 5, 'BRICKFACE': 6}
iono_label_mapper = {'g': 0, 'b': 1}


# three map functions used on target of dataSet RobotExecution
def _robot_mapper_1(x):
    return robot_label_mapper[x]


def _robot_mapper_2(x):
    return robot2_label_mapper[x]


def _robot_mapper_4(x):
    return robot4_label_mapper[x]


def _iono_mapper(x):
    return iono_label_mapper[x]


def _list_increaser(x):
    return x + 1


def load_iris():
    """
    simply load the iris data
    :return: tuple(data, target)
    """
    print ('Loading Iris....')
    iris = datasets.load_iris()
    return iris.data, iris.target


def load_digits():
    """
        simply load the digits data
        :return: tuple(data, target)
        """
    print ('Loading Digits....')
    digits = datasets.load_digits()
    return digits.data, digits.target


def _load_label_in_first_data(filename):
    """
    load the common dataSet (labels in the first column)
    internal use only
    """
    dataset = []
    target = []
    fr = open(filename)

    for line in fr.readlines():
        cur_line = line.strip().split(',')
        target.append(int(cur_line[0]))
        cur_line.remove(cur_line[0])
        flt_line = map(float, cur_line)
        dataset.append(flt_line)

    return np.array(dataset), np.array(target)


def _load_label_in_last_data(filename):
    """
    load the common dataSet (labels in the last column)
    internal use only
    """
    dataset = []
    target = []
    fr = open(filename)

    for line in fr.readlines():
        cur_line = line.strip().split(',')
        target.append(int(cur_line[-1]))
        cur_line = cur_line[:-1]
        # curLine.remove(curLine[-1])
        flt_line = map(float, cur_line)
        dataset.append(flt_line)

    return np.array(dataset), np.array(target)


def load_movement_libras():
    """
        load the 'Movement_libras' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Loading Movement_libras...')
    dataset, target = _load_label_in_last_data('../UCI Data/Movement_libras/movement_libras.data')
    target -= 1
    return dataset, target


def load_synthetic_control():
    """
        load the 'synthetic_control' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Loading Synthetic_control...')
    dataset = []
    target = []
    fr = open('../UCI Data/Synthetic_control/synthetic_control.data')

    for line in fr.readlines():
        cur_line = line.strip().split()
        flt_line = map(float, cur_line)
        dataset.append(flt_line)

    list = [0] * 100
    for i in range(6):
        target.extend(list)
        list = map(_list_increaser, list)

    return np.array(dataset), np.array(target)


def load_robot_execution(filename):
    """
        load the 'RobotExecution' dataSet
        :return: list(dataSet, target)
    """
    print ('Loading RobotExecution...')
    dataSet = []
    target = []
    row = []
    fr = open(filename)
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


def load_segmentation(normalized=True):
    target = []
    dataset = []
    fr = open('UCI Data/segmentation/segmentation.txt')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(segmentation_mapper[curLine[0]])
        fltline = map(float, curLine[1:])
        dataset.append(fltline)
    dataset = np.array(dataset)
    target = np.array(target)
    if normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)
    return dataset, target


def load_wdbc(normalized=True):
    target = []
    dataset = []
    fr = open('UCI Data/WDBC/wdbc.data.txt')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        curLine = curLine[1:]
        target.append(0 if curLine[0] == 'B' else 1)
        fltline = map(float, curLine[1:])
        dataset.append(fltline)
    dataset = np.array(dataset)
    target = np.array(target)
    if normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)
    return dataset, target


def loadRobotExecution_1():
    """
        load the 'lp1' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Loading RobotExecution_1...')
    dataSet, target = load_robot_execution('../UCI Data/RobotExecution/lp1.data')
    target = map(_robot_mapper_1, target)

    return np.array(dataSet), np.array(target)


def loadRobotExecution_2():
    """
        load the 'lp2' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Loading RobotExecution_2...')
    dataSet, target = load_robot_execution('../UCI Data/RobotExecution/lp2.data')
    target = map(_robot_mapper_2, target)
    return np.array(dataSet), np.array(target)


def loadRobotExecution_4():
    """
        load the 'lp4' dataSet
        :return: tuple(dataSet, target)
    """
    print ('Loading RobotExecution_4...')
    dataSet, target = load_robot_execution('../UCI Data/RobotExecution/lp4.data')
    target = map(_robot_mapper_4, target)
    return np.array(dataSet), np.array(target)


def loadGlass():
    """
        load glass dataset
    :return:
    """
    print ('Loading Glass...')
    dataSet, target = _load_label_in_last_data('../UCI Data/glass/glass.data.txt')
    return dataSet[:, 1:], target - 1


def loadWine():
    """
    load wine dataset
    :return:
    """
    dataSet, target = _load_label_in_first_data('../UCI Data/wine/wine.data.txt')
    return dataSet, target - 1


def loadIonosphere():
    """
    load ionosphere
    :return:
    """
    print ('Loading ionosphere...')
    dataSet = []
    target = []
    fr = open('../UCI Data/ionosphere/ionosphere.data.txt')

    for line in fr.readlines():
        curLine = line.strip().split(',')
        target.append(curLine[-1])
        curLine = curLine[:-1]
        # curLine.remove(curLine[-1])
        fltLine = map(float, curLine)
        dataSet.append(fltLine)

    target = map(_iono_mapper, target)

    return np.array(dataSet), np.array(target)


def loadIsolet():
    """
            load isolet dataset
        :return:
        """
    print ('Loading ISOLET...')
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


def load_spam_base(normalized=True):
    data, target = _load_label_in_last_data('../UCI Data/spam/spambase.data')
    data = data[:, 0:-3]
    if normalized:
        data = preprocessing.normalize(data, norm='l2')
    return data, target


def load_coil20():
    data = sio.loadmat('UCI Data/COIL20/COIL20.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    return fea, labels


def load_digit():
    data = []
    target = []
    fr = open('UCI Data/OptDigits/optdigits.tra')
    for line in fr.readlines():
        cur = line.strip().split(',')
        target.append(int(cur[-1]))
        cur = cur[:-1]
        flt_line = map(float, cur)
        data.append(flt_line)

    data = np.array(data)
    target = np.array(target)

    return data, target


def load_mnist_4000():
    data = sio.loadmat('UCI Data/MNIST_4000/2k2k.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    labels[labels == 255] = 9
    return fea, labels


def load_mnist_full():
    data = sio.loadmat('UCI Data/MNIST_FULL/Orig.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    labels[labels == 255] = 9
    return fea, labels


def load_usps():
    data = sio.loadmat('UCI Data/USPS/USPS.mat')
    fea = data['fea']
    fea = fea.astype(np.float64)
    labels = data['gnd'] - 1
    labels = labels.flatten()
    # labels[labels == 255] = 9
    return fea, labels


def load_tr23(sparse=False):

    return


def load_wap(sparse_type='dense'):
    rows = np.empty(0, dtype=int)
    cols = np.empty(0, dtype=int)
    vals = np.empty(0, dtype=float)
    with open('../UCI Data/wap/wap.mat') as f:
        flag = True
        count = 0
        for line in f:
            if flag:
                flag = False
                continue
            elements = line.split(' ')
            elements = [x for x in elements if x != '']
            idx = elements[0::2]
            values = elements[1::2]
            if len(idx) != len(values):
                raise Exception('index no eq to values')
            rows = np.hstack([rows, np.full(len(idx), count, dtype=int)])
            cols = np.hstack([cols, np.array(idx, dtype=int) - 1])
            vals = np.hstack([vals, np.array(values, dtype=float)])
            count += 1
    all_class = []
    labels = []
    with open('../UCI Data/wap/wap.mat.rclass') as label_f:
        for line in label_f:
            if line == '':
                continue
            if line in all_class:
                labels.append(all_class.index(line))
            else:
                all_class.append(line)
                labels.append(all_class.index(line))
    labels = np.array(labels, dtype=int)
    wap_data = coo_matrix((vals, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))
    if sparse_type == 'coo':
        pass
    elif sparse_type == 'csr':
        wap_data = wap_data.tocsr()
    elif sparse_type == 'csc':
        wap_data = wap_data.tocsc()
    else:
        wap_data = wap_data.toarray()
    return wap_data, labels


def load_musk_2_data(normalized=True):
    fr = open('UCI Data/MUSK/Musk-2.data')
    count = 0
    feature_vectors = []
    target = []
    for line in fr.readlines():
        count += 1
        line_elements = line.strip().split(',')
        feature = line_elements[2:-1]
        feature = map(float, feature)
        feature_vectors.append(feature)
        target.append(0 if line_elements[-1] == '0.' else 1)
        print str(count) + " -- " + str(len(line_elements)) + "  " + line_elements[-1]
    target = np.array(target)
    feature_vectors = np.array(feature_vectors)
    print feature_vectors.shape
    if normalized:
        feature_vectors = preprocessing.normalize(feature_vectors, norm='l2')
        # min_max_scaler = preprocessing.MinMaxScaler()
        # feature_vectors = min_max_scaler.fit_transform(feature_vectors)
        return feature_vectors, target
    # print np.sum(target)
    # print target
    return feature_vectors, target


def load_sat(normalized=True):
    fr = open('UCI Data/SAT/sat.data')
    count = 0
    feature_vectors = []
    target = []
    for line in fr.readlines():
        count += 1
        line_elements = line.strip().split(' ')
        feature = line_elements[0:-1]
        feature = map(float, feature)
        feature_vectors.append(feature)
        target.append(5 if line_elements[-1] == '7' else (int(line_elements[-1]) - 1))
        print str(count) + " -- " + str(len(line_elements)) + "  " + line_elements[-1]
    target = np.array(target)
    feature_vectors = np.array(feature_vectors)
    print feature_vectors.shape
    if normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        feature_vectors = min_max_scaler.fit_transform(feature_vectors)
    # print np.sum(target)
    # print target
    return feature_vectors, target


def load_skin(normalized=True):
    dataset = []
    target = []
    fr = open('UCI Data/Skin/Skin_NonSkin.txt')

    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        target.append(int(cur_line[-1]))
        cur_line = cur_line[:-1]
        # curLine.remove(curLine[-1])
        flt_line = map(float, cur_line)
        dataset.append(flt_line)
    dataset = np.array(dataset)
    target = np.array(target)
    if normalized:
        data_normed = preprocessing.normalize(dataset, norm='l2')
        return data_normed, target
        # min_max_scaler = preprocessing.MinMaxScaler()
        # dataset = min_max_scaler.fit_transform(dataset)

    return dataset, target


def load_covtype():
    data = np.loadtxt('covtype.data', delimiter=',')
    print data.shape
    targets = data[:, -1].flatten() - 1
    print np.unique(targets)
    print targets.shape
    data = data[:, :-1]
    print data.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data)
    return X_train_minmax, targets


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

    # dataSet, target = loadIsolet()

    dataSet, target = load_sat()
    print dataSet
    print target
    print (dataSet.shape)
    print (target.shape)


if __name__ == '__main__':
    Test()
