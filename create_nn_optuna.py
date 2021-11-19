import os

import numpy as np
import pickle
import sys
import random
from pathlib import Path
import oneHot_deep
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from plan_generator import PlanGenerator
from params_generator import ParamsGenerator

matplotlib.use('agg')
from sklearn import metrics
import getopt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU
#from AttentionMechanism import AttentionL
from attention_extraction_layers import AttentionWeights, ContextVector
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.losses import Loss, BinaryCrossentropy
from os.path import join
from utils_functions import load_files, create_table, create_plot, load_from_pickles
from typing import Union
from tensorflow.keras.models import load_model

import optuna
from optuna.samplers import TPESampler



def build_network_single_fact(generator: PlanGenerator,
                              embedding_params: dict = None,
                              hidden_layers: int = 1,
                              regularizer_params: dict = None,
                              recurrent_list: list = ['lstm', None],
                              use_attention: bool = True,
                              optimizer_list: list = ['adam', None],
                              loss_function: Union[Loss, str] = 'binary_crossentropy',
                              model_name: str = 'model') -> Model:
    inputs = Input(shape=(generator.max_dim,))
    prev_layer = inputs

    if embedding_params is not None:
        embedding_layer = Embedding(input_dim=len(generator.dizionario)+1,
                                    input_length=generator.max_dim,
                                    **embedding_params)(prev_layer)
        prev_layer = embedding_layer

    
    for layer in range(hidden_layers):

        if regularizer_params is None or (regularizer_params['l1'] is None and regularizer_params['l2'] is None):
            regularizer = None
        elif regularizer_params['l1'] is None:
            regularizer = l2(regularizer_params['l2'])
        elif regularizer_params['l2'] is None:
            regularizer = l1(regularizer_params['l1'])
        else:
            regularizer = l1_l2(l1=regularizer_params['l1'], l2=regularizer_params['l2'])

        recurrent_type, recurrent_params = recurrent_list
        if recurrent_type == 'lstm':
            recurrent_layer = LSTM(**recurrent_params, activity_regularizer= regularizer,
                                   name=f'lstm_layer_{layer}')(prev_layer)
        elif recurrent_type == 'gru':
            recurrent_layer = GRU(**recurrent_params, activity_regularizer= regularizer,
                                  name=f'gru_layer_{layer}')(prev_layer)
        prev_layer = recurrent_layer

    if use_attention:
        attention_weights = AttentionWeights(generator.max_dim, name='attention_weights')(prev_layer)
        context_vector = ContextVector()([prev_layer, attention_weights])
        prev_layer = context_vector

    outputs = Dense(len(generator.dizionario_goal), activation='softmax', name='output')(prev_layer)

    optimizer_type, optimizer_params = optimizer_list
    if optimizer_type == 'adam':
        optimizer = Adam(**optimizer_params)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model


def get_embedding_attention_network_params(model_name: str,
                                           embedding_dim: int = 124,
                                           recurrent_type: str = 'lstm',
                                           recurrent_dim: int = 64,
                                           dropout: float = 0,
                                           recurrent_dropout: float = 0,
                                           learning_rate: float = 0.001,
                                           loss_function: Union[str, Loss] = 'binary_crossentropy') -> dict:
    to_ret = {
        'embedding_params': {
            'output_dim': embedding_dim,
            'mask_zero': True},

        'recurrent_list': [
            recurrent_type,
            {'units': recurrent_dim,
             'activation': 'linear',
             'return_sequences': True,
             'dropout': dropout,
             'recurrent_dropout': recurrent_dropout}],

        'use_attention': True,

        'optimizer_list': [
            'adam',
            {'lr': learning_rate,
             'beta_1': 0.9,
             'beta_2': 0.999,
             'amsgrad': False,
             'clipnorm': 1.0}
        ],

        'loss_function': loss_function,

        'model_name': model_name
    }
    return to_ret
    
def get_default_params():
    p = ParamsGenerator('multi_goal',
                        recurrent_type='lstm',
                        output_dim=60,
                        units=128,
                        loss_function='binary_crossentropy'
                        )
    return p.generate(1)[0]

def get_loss_name(loss : Union[str, Loss]):
    if type(loss) is not str:
        loss = str(loss)
        loss = loss.rsplit('.', 1)[1]
        loss = loss.split()[0]

    return loss


def print_network_details(model: Model, params: dict, save_file: str = None) -> None:
    headers = ['EMBEDDING DIM', 'LOSS FUNCTION', 'RECURRENT DIM', 'DROPOUT', 'RECURRENT_DROPOUT']

    rows = [
        [params['embedding_params']['output_dim'],
         get_loss_name(params['loss_function']),
         params['recurrent_list'][1]['units'],
         params['recurrent_list'][1]['dropout'],
         params['recurrent_list'][1]['recurrent_dropout']]]
    title = f'{model.name} details'
    to_print = create_table(title, headers, rows, just=18)
    to_print.append(model.summary())
    if save_file is None:
        for line in to_print:
            print(line)
    else:
        with open(save_file, 'w') as f:
            for line in to_print:
                f.write(line)
            f.close()

def train_network(model: Model,
                  train_generator: PlanGenerator,
                  epochs: int = 3,
                  verbose: int = 2,
                  callbacks: list = None,
                  validation_generator: PlanGenerator = None) -> dict:

    history = model.fit(x=train_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks)
    return history

def save_plot(history : dict, plot_dir : str = None) -> None:
    y = []
    keys = [['train', 'loss'], ['validation', 'val_loss']]
    for label, k in keys:
        y.append([label, history.history[k]])
    x = None
    create_plot(plot_type='simple',
                target_dir=plot_dir,
                x=x,
                y=y,
                xlabel='Epochs',
                ylabel='Loss')


def print_metrics(y_true: list, y_pred: list, dizionario_goal: dict, save_dir: str = None, filename: str = 'metrics') -> list:
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]
    labels = list(dizionario_goal.keys())
    to_print = []
    accuracy = metrics.accuracy_score(y_true, y_pred)
    hamming_loss = metrics.hamming_loss(y_true, y_pred)
    to_print.append(f'Accuracy: {accuracy}\n')
    to_print.append(f'Hamming Loss: {hamming_loss}\n')
    to_print.append(metrics.classification_report(y_true, y_pred, target_names=labels))
    if save_dir is None:
        for line in to_print:
            print(line)
    else:
        with open(join(save_dir, f'{filename}.txt'), 'w') as file:
            for line in to_print:
                file.write(line)
            file.close()
    return [accuracy, hamming_loss]


def get_model_predictions(model: Model, test_generator: PlanGenerator) -> list:
    y_pred = list()
    y_true = list()
    for i in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(i)
        y_pred.extend(model.predict(x))
        y_true.extend(y)
    return y_pred, y_true




def get_callback_default_params(callback_name: str) -> list:
    to_return = dict()
    if callback_name == 'model_checkpoint':
        to_return['save_weights_only'] = False
        to_return['monitor'] = 'val_loss'
        to_return['mode'] = 'auto'
        to_return['save_best_only'] = True
    if callback_name == 'early_stopping':
        to_return['monitor'] = 'val_loss'
        to_return['verbose'] = 2
        to_return['patience'] = 5
        to_return['mode'] = 'min'
        to_return['restore_best_weights'] = True
    return to_return

def run_tests(test_plans: list, dizionario: dict, dizionario_goal: dict, batch_size: int, max_plan_dim:int,
              plan_percentage: float, save_dir: str, filename='metrics') -> None:
    if test_plans is not None:
        test_plans = test_plans[0]
        test_generator = PlanGenerator(test_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
                                       plan_percentage, shuffle=False)
        y_pred, y_true = get_model_predictions(test_generator)
        scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=dizionario_goal, save_dir=save_dir, filename=filename)

def objective(trial: optuna.Trial,
              model_name: str,
              train_plans: list,
              val_plans: list,
              dizionario: dict,
              dizionario_goal: dict,
              max_plan_dim: int,
              plan_percentage: float):


    if train_plans is not None and dizionario is not None and dizionario_goal is not None:
        train_generator = PlanGenerator(train_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
                                        plan_percentage)
        val_generator = PlanGenerator(val_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
                                          plan_percentage, shuffle=False)

    use_dropout = trial.suggest_categorical('use_dropout', [True, False])
    if use_dropout:
        dropout = trial.suggest_uniform('dropout', 0, 0.5)
    else:
        dropout = 0
    use_recurrent_dropout = trial.suggest_categorical('use_recurrent_dropout', [True, False])

    if use_recurrent_dropout:
        recurrent_dropout = trial.suggest_uniform('recurrent_dropout', 0, 0.5)
    else:
        recurrent_dropout = 0

    use_activity_regularisation = trial.suggest_categorical('use_activity_regularisation', [True, False])
    if use_activity_regularisation:
        l1 = trial.suggest_categorical('l1', [0.01, 0.001, 0.0001, 0.00001])
        l2 = trial.suggest_categorical('l2', [0.01, 0.001, 0.0001, 0.00001])
    else:
        l1 = None
        l2 = None

    params = ParamsGenerator(model_name=model_name,
                             recurrent_type= 'lstm',
                             units=trial.suggest_int('hidden_layer_dim', 56, 256),
                             output_dim=trial.suggest_int('embedding_dim', 30, 120),
                             dropout= dropout,
                             recurrent_dropout=recurrent_dropout,
                             l1=l1,
                             l2=l2)

    params = params.generate(1)[0]

    model = build_network_single_fact(train_generator, **params)
    print_network_details(model, params)
    callbacks = [
        EarlyStopping(**get_callback_default_params('early_stopping')),
    ]
    history = train_network(model=model,
                            train_generator=train_generator,
                            callbacks=callbacks,
                            validation_generator=val_generator,
                            epochs=epochs)

    y_pred, y_true = get_model_predictions(model, val_generator)
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]
    result = metrics.f1_score(y_true, y_pred, average='micro')
    return result

if __name__ == '__main__':
    np.random.seed(47)

    read_plans_dir = 'dataset_whole2_small'
    read_dict_dir = 'dataset_whole2_small'
    target_dir = 'modello_prova'
    plan_percentage = 0.5
    batch_size = 64
    max_plan_dim = 100
    
    
    epochs = 1
    lr = 0.001
    log_dir = './'
    loss_function = 'binary_crossentropy'
    compute_model = True
    compute_results = False
    incremental_tests = False
    params_dir = None

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', ['read-plans-dir=', 'target-dir=', 'plan-perc=', 'batch-size=', 'max-plan-dim=',
                                          'log-dir=', 'model-name=', 'epochs=', 'read-dict-dir='])
                                          
    for opt, arg in opts:
        if opt == "--read-plans-dir":
            read_plans_dir = arg
        if opt == "--read-dict-dir":
            read_dict_dir = arg
        elif opt == "--target-dir":
            target_dir = arg
        elif opt == "--plan-perc":
            plan_percentage = float(arg)
        elif opt == "--batch-size":
            batch_size = int(arg)
        elif opt == "--epochs":
            epochs = int(arg)
        elif opt == '--max-plan-dim':
            max_plan_dim = int(arg)
        elif opt == '--log-dir':
            log_dir = arg
        elif opt == '--model-name':
            model_name = arg
            
#    if params_dir != None:
#        params = load_files(params_dir,
#                             load_ok=f'Params loaded from {params_dir}',
#                             error=f'Could not load params from {params_dir}.\nCheck the --params-dict option.')
#    else:
#        params = get_default_params()

    study_name = f'{model_name}'
    n_trials=30
    db_dir = '/data/users/mchiari/goal_recognition/optuna_studies/'

    [dizionario, dizionario_goal] = load_from_pickles(read_dict_dir, ['dizionario', 'dizionario_goal'])
    [train_plans, val_plans] = load_from_pickles(read_plans_dir, ['train_plans', 'val_plans'])

    max_plan_dim = int(plan_percentage * max_plan_dim )

    study = optuna.create_study(
        storage=f'sqlite:///{join(db_dir, f"{study_name}.db")}',
        sampler= TPESampler(seed=43),
        direction='maximize',
        load_if_exists=True,
        study_name=study_name
    )
    study.optimize(
        lambda trial: objective(trial=trial,
                                dizionario= dizionario,
                                dizionario_goal= dizionario_goal,
                                model_name=model_name,
                                train_plans=train_plans,
                                val_plans=val_plans,
                                max_plan_dim=max_plan_dim,
                                plan_percentage=plan_percentage),
        n_trials=n_trials,
        gc_after_trial=True
    )


    plot_dir = join(target_dir, study_name)
    plot_dir = join(plot_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(join(plot_dir, 'opt_hist.png'))

    fig = optuna.visualization.plot_slice(study)
    fig.write_image(join(plot_dir, 'slice.png'))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(join(plot_dir, 'param_importance.png'))
    
#    hidden_layer_dim = params['recurrent_list'][1]['units']
#    embedding_dim = params['embedding_params']['output_dim']
#    dropout = params['recurrent_list'][1]['dropout']
#    recurrent_dropout = params['recurrent_list'][1]['recurrent_dropout']
#    model_name = params['model_name']
#    recurrent_type = params['recurrent_list'][0]
#    loss_function = params['loss_function']
    
#    if loss_function == 'SigmoidFocalCrossEntropy':
#        params['loss_function'] = SigmoidFocalCrossEntropy()

#    max_plan_dim = int(plan_percentage * max_plan_dim)
#    model_dir_name = f'{model_name}_{recurrent_type}_epochs={epochs}_embedding={embedding_dim}_units={hidden_layer_dim}_dropout={dropout}_' \
#                     f'recurrent-dropout={recurrent_dropout}_loss={get_loss_name(loss_function)}_plan-percentage={plan_percentage}_batch-size={batch_size}'
#    model_dir = join(target_dir, model_dir_name)
#    plot_dir = join(model_dir, 'plots')
#    os.makedirs(plot_dir, exist_ok=True)


#

#    if compute_model:
#        [train_plans, val_plans] = load_from_pickles(read_plans_dir, ['train_plans', 'val_plans'])
#        if train_plans is not None and dizionario is not None and dizionario_goal is not None:
#            train_generator = PlanGenerator(train_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
#                                            plan_percentage)
#            val_generator = None
#            if val_plans is not None:
#                val_generator = PlanGenerator(val_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
#                                              plan_percentage, shuffle=False)

#            model = build_network_single_fact(train_generator, **params)
#            print_network_details(model, params)


#            callbacks = [
#                EarlyStopping(**get_callback_default_params('early_stopping')),
#                ModelCheckpoint(filepath=join(model_dir, 'checkpoint'), **get_callback_default_params('model_checkpoint'))
#            ]
#            history = train_network(model=model,
#                                    train_generator=train_generator,
#                                   callbacks=callbacks,
#                                    validation_generator=val_generator,
#                                    epochs=epochs)
#            model.save(join(model_dir, f'{model_name}.h5'))
#            save_plot(history=history, plot_dir=join(plot_dir, 'loss_plot.png'))

#    else:
#        model = load_model(join(model_dir, f'{model_name}.h5'),
#                           custom_objects={'AttentionWeights': AttentionWeights,'ContextVector': ContextVector})
#        print(model.summary())

#    if compute_results:
#        if incremental_tests:
#            test_dir = join(read_plans_dir, 'incremental_test_sets')
#            files = os.listdir(test_dir)
#        else:
#            test_dir = read_plans_dir
#            files = ['test_plans']

#        for f in files:
#            test_plans = load_from_pickles(test_dir, [f])
#            if not(test_plans is None) and len(test_plans)>0:
#                run_tests(test_plans=test_plans, dizionario=dizionario, dizionario_goal=dizionario_goal, batch_size=batch_size,
#                          max_plan_dim=max_plan_dim, plan_percentage=plan_percentage, save_dir=model_dir, filename=f'metrics_{f}')
#            else:
#                print(f'Problems with file {f} in folder {test_dir}')


