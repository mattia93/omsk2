
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