from easydict import EasyDict as edict
import numpy as np
__C = edict()
cfg = __C

#
# Dataset Config
#
# __C.DATASETS = edict()
__C.HeartFailure = edict()

__C.HeartFailure.PATH = '../Dataset/heart.csv'

__C.HeartFailure.LOGS_LOSSES_PATH = '../Logs/Losses'
__C.HeartFailure.LOGS_PLOTS_PATH = '../Logs/Plots'
__C.HeartFailure.LOGS_REPORTS_PATH = '../Logs/Reports'

__C.HeartFailure.MODELS_PATH = '../Models'

__C.HeartFailure.TASK_NAME = 'HeartFailure'
__C.HeartFailure.MODEL_NAME = 'DecisionTree'

__C.HeartFailure.NORMALIZER = 'MinMaxScaler'
__C.HeartFailure.Encoder = 'OneHotEncoder'

__C.HeartFailure.SPLIT = 0.2
__C.HeartFailure.RANDOM_STATE = 42

__C.HeartFailure.COLUMNS = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
                            'HeartDisease']

__C.HeartFailure.TARGET = ['HeartDisease']


__C.HeartFailure.CATEGORICAL_COLUMNS = [
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

__C.HeartFailure.NUMERICAL_COLUMNS = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']


#! Support Vector Machine
__C.HeartFailure.SVM_parameters_grid = edict()
__C.HeartFailure.SVM_parameters_grid.C = [0.01, 0.1, 1, 10, 100]
__C.HeartFailure.SVM_parameters_grid.gamma = [
    'scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
__C.HeartFailure.SVM_parameters_grid.kernel = [
    'linear', 'poly', 'rbf', 'sigmoid']
__C.HeartFailure.SVM_parameters_grid.class_weight = ['balanced', None]


#! Decision Tree
__C.HeartFailure.DT_parameters_grid = edict()

__C.HeartFailure.DT_parameters_grid.max_depth = [
    3, 5, 10, None]
__C.HeartFailure.DT_parameters_grid.min_samples_split = [2, 5, 10]
# __C.HeartFailure.DT_parameters_grid.MIN_SAMPLES_LEAF = [1, 2, 4]
# __C.HeartFailure.DT_parameters_grid.MAX_FEATURES = ['auto', 'sqrt', 'log2']
__C.HeartFailure.DT_parameters_grid.criterion = ['gini', 'entropy']


#! Random Forest
__C.HeartFailure.RF_parameters_grid = edict()
__C.HeartFailure.RF_parameters_grid.n_estimators = [
    int(x) for x in range(100, 1200, 100)]
__C.HeartFailure.RF_parameters_grid.max_features = ['auto', 'sqrt']
__C.HeartFailure.RF_parameters_grid.max_depth = [
    int(x) for x in range(10, 110, 10)]
__C.HeartFailure.RF_parameters_grid.min_samples_split = [2, 5, 10]
__C.HeartFailure.RF_parameters_grid.min_samples_leaf = [1, 2, 4]
__C.HeartFailure.RF_parameters_grid.bootstrap = [True, False]


#! K-Nearest Neighbors
__C.HeartFailure.KNN_parameters_grid = edict()
__C.HeartFailure.KNN_parameters_grid.n_neighbors = list(range(1, 16))
__C.HeartFailure.KNN_parameters_grid.weights = ['uniform', 'distance']
__C.HeartFailure.KNN_parameters_grid.metric = [
    'euclidean', 'manhattan', 'minkowski']
__C.HeartFailure.KNN_parameters_grid.algorithm = [
    'auto', 'ball_tree', 'kd_tree', 'brute']

#! Multi-Layer Perceptron
__C.HeartFailure.MLP_parameters_grid = edict()

#! Gaussian Naive Bayes
__C.HeartFailure.GaussBayes_parameters_grid = edict()
__C.HeartFailure.GaussBayes_parameters_grid.var_smoothing = np.logspace(
    -9, 0, 500)


#! Categorical Naive Bayes
__C.HeartFailure.CatBayes_parameters_grid = edict()
__C.HeartFailure.CatBayes_parameters_grid.alpha = [0.01, 0.1, 0.5, 1.0, 10.0]
__C.HeartFailure.CatBayes_parameters_grid.fit_prior = [True, False]


#! Bernoulli Naive Bayes
__C.HeartFailure.bernoulliBayes_parameters_grid = edict()
__C.HeartFailure.bernoulliBayes_parameters_grid.alpha = [
    0.01, 0.1, 0.5, 1.0, 10.0]
__C.HeartFailure.bernoulliBayes_parameters_grid.fit_prior = [True, False]
__C.HeartFailure.bernoulliBayes_parameters_grid.binarize = [
    None, 0.0, 0.5, 1.0]


__C.HeartFailure.Hierarchical_parameters_grid = edict()
__C.HeartFailure.Hierarchical_parameters_grid.n_clusters = [2, 3, 4, 5, 6]
__C.HeartFailure.Hierarchical_parameters_grid.metric = [
    'euclidean', 'l1', 'l2', 'manhattan', 'cosine']
__C.HeartFailure.Hierarchical_parameters_grid.linkage = [
    'ward', 'complete', 'average', 'single']

#! XGBoost
__C.HeartFailure.XGB_parameters_grid = edict()

#! Logistic Regression
__C.HeartFailure.LR_parameters_grid = edict()

#! AdaBoost
__C.HeartFailure.ADA_parameters_grid = edict()


# __C.HeartFailure.MODELS_PATH = '/home/hzxie/Datasets/HeartFailure/HeartFailureVox32/%s/%s/model.binvox'


#
# Dataset
#
# __C.DATASET = edict()
# __C.DATASET.TRAIN_DATASET = 'HeartFailure'
# __C.DATASET.TEST_DATASET = 'HeartFailure'

#
# Common
#
# __C.CONST = edict()
# __C.CONST.DEVICE = '0'

# __C.CONST.BATCH_SIZE = 64


# Network
#
# __C.NETWORK = edict()
# __C.NETWORK.LEAKY_VALUE = .2

#
# Training
#
# __C.TRAIN = edict()
# __C.TRAIN.NUM_WORKER = 4             # number of data workers
# __C.TRAIN.NUM_EPOCHES = 250
# __C.TRAIN.POLICY = 'adam'        # available options: sgd, adam
