
class ERRORS:
    MSG_ERROR_LOAD_PLANS = f'Error while loading the plans'
    STD_ERROR_LOAD_FILE = 'Error while loading {0}'
    STD_LOAD_FILE_OK = '{0} loaded from {1}'
    STD_FILE_NOT_SAVED = '{0} was not saved'

class CREATE_TRAIN_TEST:
    PLANS_NUMBER = 'Total plans : {0}'
    GOALS_NUMBER = 'Total goals: {0}'
    ACTIONS_NUMBER = 'Total actions: {0}'
    TRAIN_PLANS_NUMBER = 'Total train plans: {0}'
    TEST_PLANS_NUMBER = 'Total test plans: {0}'
    VALIDATION_PLANS_NUMBER = 'Total validation plans: {0}'

class FILENAMES:
    TRAIN_PLANS_FILENAME = 'train_plans'
    VALIDATION_PLANS_FILENAME = 'val_plans'
    TEST_PLANS_FILENAME = 'test_plans'
    PLOT_ACTIONS_FILENAME = 'actions_frequency.png'
    PLOT_GOALS_FILENAME = 'goals_frequency.png'
    PLOT_LENGTH_FILENAME = 'plans_length.png'
    PLANS_FILENAME = 'plans'
    ACTION_DICT_FILENAME = 'dizionario'
    GOALS_DICT_FILENEME = 'dizionario_goal'

class CREATE_DATASET:
    GOALS_NUMBER = 'Total goals: {0}'

class HELPS:
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
