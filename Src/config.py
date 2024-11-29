from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
# __C.DATASETS = edict()
__C.HeartFailure = edict()

__C.HeartFailure.PATH = '../Dataset/dataset.csv'

__C.HeartFailure.LOGS_LOSSES_PATH = '../Logs/Losses'
__C.HeartFailure.LOGS_PLOTS_PATH = '../Logs/Plots'

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

#
# Directories
#
# __C.DIR = edict()
# __C.DIR.OUT_PATH = './output'
# __C.DIR.RANDOM_BG_PATH = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
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


#
# Testing options
#
