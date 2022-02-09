
class ERRORS:
    MSG_ERROR_LOAD_PARAMS = 'Error while loading the parameters'
    MSG_ERROR_LOAD_PLANS = f'Error while loading the plans'
    STD_ERROR_LOAD_FILE = 'Error while loading {0}'
    STD_LOAD_FILE_OK = '{0} loaded from {1}'
    STD_FILE_NOT_SAVED = '{0} was not saved'

class CREATE_TRAIN_TEST:
    TRAIN_PLANS_NUMBER = 'Total train plans: {0}'
    TEST_PLANS_NUMBER = 'Total test plans: {0}'
    VALIDATION_PLANS_NUMBER = 'Total validation plans: {0}'

class FILENAMES:
    STATS_FILENAME = 'stats.txt'
    TRAIN_PLANS_FILENAME = 'train_plans'
    VALIDATION_PLANS_FILENAME = 'val_plans'
    TEST_PLANS_FILENAME = 'test_plans'
    PLOT_ACTIONS_FILENAME = 'actions_frequency.png'
    PLOT_GOALS_FILENAME = 'goals_frequency.png'
    PLOT_LENGTH_FILENAME = 'plans_length.png'
    PLANS_FILENAME = 'plans'
    ACTION_DICT_FILENAME = 'dizionario'
    GOALS_DICT_FILENEME = 'dizionario_goal'
    NETWORK_PLOTS_FOLDER = 'plots'

class CREATE_DATASET:
    GOALS_TABLE_TITLE = 'Goals frequency'
    ACTIONS_TABLE_TITLE = 'Actions frequency'
    PLANS_TABLE_TITLE = 'Plans length'
    PLANS_NUMBER = 'Total plans : {0}'
    GOALS_NUMBER = 'Total goals: {0}'
    ACTIONS_NUMBER = 'Total actions: {0}'
    TABLE_HEADERS = ['MIN', 'Q1', 'Q2', 'Q3', 'MAX']
    NBINS = 50

class HELPS:
    SAVE_STATS_FLAG = 'Flag for creating the file stats.txt that saves the stats in the plots directory'
    PRED_DIR_SRC = 'Folder that contains the prediction file(s)'
    PRED_DIR_OUT = 'Folder where to save the predictions.'
    PLAN_PERCENTAGE = 'Percentage of actions per plan.'
    TEST_PLANS_DIR_SRC = 'Folder that contains the test plans.'
    MODEL_SRC = 'Path of the model file to use.'
    MODEL_NAME = 'Name of your model; it will also be the name of the Optuna Study.'
    DB_DIR = 'Folder that contains the database where to store the Optuna Study.'
    TRIALS = 'Number of Optuna trials.'
    INCREMENTAL_TESTS_FLAG = 'Flag for incremental tests.'
    PLANS_FOLDER_SRC = 'Folder that contains the train, test and validation plans.'
    MIN_PLAN_PERCENTAGE = 'Minimum percentage of actions per plan considered.'
    EPOCHS = 'Number of epochs for training the network.'
    DICT_FOLDER_SRC = 'Folder that contains the actions and goals dictionaries.'
    BATCH_SIZE = 'Size of each batch'
    MAX_PLAN_PERCENTAGE = 'Maximum percentage of action per plan considered.'
    MODEL_DIR_OUT = 'Folder where to save the model directory.'
    NETWORK_PARAMETERS_SRC = 'Path to the network parameters file.'
    CREATE_IF_NOT_EXISTS = "It's created if it does not exists."
    XML_FOLDER_SRC = 'Folder that contains the XMLs files.'
    PLANS_AND_DICT_FOLDER_OUT = "Folder where to store plans and dictionaries file."
    ONEHOT_FLAG = 'Flag that applies the one-hot representation for the goals.'
    PLOTS_FOLDER_OUT = 'Folder where to save the plots.'
    TRAIN_TEST_VAL_FOLDER_OUT = 'Folder where to save the train, test and validation files.'
    PLANS_AND_DICT_FOLDER_SRC = 'Folder that contains the plans and dictionaries pickles.'
    MAX_PLAN_LENGTH = 'Maximum plan length accepted.'
    TRAIN_PERCENTAGE = 'Percentage of plans used to create the training set.'
    NO_VAL_FLAG = 'Flag used not to create the validation set'


class KEYS:
    MODEL_NAME = 'MODEL_NAME'
    STUDY = 'STUDY'
    TARGET_DIR = 'TARGET_DIR'
    MAX_PLAN_DIM = 'MAX_PLAN_DIM'
    MIN_PLAN_PERC = 'MIN_PLAN_PERC'
    MAX_PLAN_PERC = 'MAX_PLAN_PERC'
    READ_PLANS_DIR = 'READ_PLANS_DIR'
    BATCH_SIZE = 'BATCH_SIZE'
    EPOCHS = 'EPOCHS'
    GOALS_DICT = 'GOALS_DICT'
    ACTION_DICT = 'ACTION_DICT'
    MODEL_DIR = 'MODEL_DIR'
    PARAMS = 'PARAMS'
    PLANS = 'PLANS'

