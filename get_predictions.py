import getopt
import pickle
import sys

import numpy as np
from keras.models import load_model
#from AttentionMechanism import AttentionL
from attention_extraction_layers import AttentionWeights, ContextVector
from plan_generator import PlanGenerator
from utils_functions import load_files, load_from_pickles
import os
from os.path import join

def get_predictions(generator : PlanGenerator, model, batch_size):
    y_true = []
    y_pred = []
    for i in range(generator.__len__()):
        x, y = generator.__getitem__(i)
        y_true.extend(y)
        y_pred.extend(model.predict(x, batch_size=batch_size))
    return y_true, y_pred





if __name__ == '__main__':
    np.random.seed(47)
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', ['read-test-plans-dir=', 'read-dict-dir=', 'model-file=', 'target-dir=', 
                                          'plan-percentage=', 'max-plan-dim=', 'batch-size='])
    read_dict_dir = ''
    target_dir = '../results' 
    read_test_plans_dir = ''
    model_file = ''
    plan_percentage = 0.5
    batch_size = 64
    max_plan_dim = 100

    for opt, arg in opts:
        if opt == '--read-dict-dir':
            read_dict_dir = arg
        elif opt == '--read-test-plans-dir':
            read_test_plans_dir = arg
        elif opt=='--model-file':
            model_path = arg 
        elif opt == '--target-dir':
            target_dir = arg
        elif opt == "--plan-percentage":
            perc_action = float(arg)
        elif opt == "--batch-size":
            batch_size = int(arg)
        elif opt == '--max-plan-dim':
            max_dim = int(arg)

    
    try:
        model = load_model(model_path,  custom_objects={'AttentionWeights': AttentionWeights,
                                                                       'ContextVector' : ContextVector})
        print('Model loaded')
        print(model.summary())
    except OSError as e:
        print(e)
        print('Error while loading the model.\n'
              'Please check the -r parameter is correct')
        model = None
        
    
    
    filenames = os.listdir(read_test_plans_dir)
    [dizionario, dizionario_goal] = load_from_pickles(read_dict_dir, ['dizionario', 'dizionario_goal'])
    test_plans = load_from_pickles(read_test_plans_dir, filenames)

    max_dim = int(perc_action*max_dim)

    if model is None or test_plans is None or dizionario is None or dizionario_goal is None:
        print('Could not create the file')
    else:
        for i, plans in enumerate(test_plans):
            gen = PlanGenerator(plans, dizionario, dizionario_goal, batch_size, max_dim, perc_action, shuffle=False)
            filename = filenames[i]
            y_true, y_pred = get_predictions(gen, model, batch_size)
            res_dict = dict()
            res_dict['y_test_true'] = y_true
            res_dict['y_test_pred'] = y_pred

            os.makedirs(target_dir, exist_ok=True)
            with open(join(target_dir, filename + '.pkl'), 'wb') as file:
                pickle.dump(res_dict, file)
                print(f'Saved {filename}.pkl file in {target_dir}')