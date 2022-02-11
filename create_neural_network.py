import os
import click
import numpy as np
from constants import HELPS, ERRORS, KEYS, FILENAMES
import matplotlib
from plan_generator import PlanGeneratorMultiPerc
from params_generator import ParamsGenerator
matplotlib.use('agg')
from sklearn import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU, Bidirectional
from attention_extraction_layers import AttentionWeights, ContextVector
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.losses import Loss
from os.path import join
from utils_functions import load_file, create_table, create_plot, load_from_pickles
from typing import Union
from tensorflow.keras.models import load_model
import optuna
from optuna.samplers import TPESampler
from tensorflow.keras.metrics import Precision, Recall


def build_network_single_fact(generator: PlanGeneratorMultiPerc,
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
        embedding_layer = Embedding(input_dim=len(generator.dizionario) + 1,
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
            recurrent_layer = LSTM(**recurrent_params, name=f'lstm_layer_{layer}')(prev_layer)
        elif recurrent_type == 'gru':
            recurrent_layer = GRU(**recurrent_params, name=f'gru_layer_{layer}')(prev_layer)
        elif recurrent_type == 'bilstm':
            lstm = LSTM(**recurrent_params)
            recurrent_layer = Bidirectional(layer=lstm)(prev_layer)
        prev_layer = recurrent_layer

    if use_attention:
        attention_weights = AttentionWeights(generator.max_dim, name='attention_weights')(prev_layer)
        context_vector = ContextVector()([prev_layer, attention_weights])
        prev_layer = context_vector

    outputs = Dense(len(generator.dizionario_goal), activation='sigmoid', name='output')(prev_layer)

    optimizer_type, optimizer_params = optimizer_list
    if optimizer_type == 'adam':
        optimizer = Adam(**optimizer_params)
    if loss_function == 'SigmoidFocalCrossEntropy':
        loss_function = SigmoidFocalCrossEntropy()
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy',Precision(name='precision'),
                                                                    Recall(name='recall')])

    return model


def print_network_details(model: Model, params: dict, save_file: str = None) -> None:
    headers = ['EMBEDDING DIM', 'LOSS FUNCTION', 'RECURRENT DIM', 'DROPOUT', 'RECURRENT_DROPOUT']

    rows = [
        [params['embedding_params']['output_dim'],
         params['loss_function'],
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
                  train_generator: PlanGeneratorMultiPerc,
                  epochs: int = 3,
                  verbose: int = 2,
                  callbacks: list = None,
                  validation_generator: PlanGeneratorMultiPerc = None) -> dict:
    history = model.fit(x=train_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks)
    return history


def save_plot(history: dict, plot_dir: str = None) -> None:
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


def print_metrics(y_true: list, y_pred: list, dizionario_goal: dict, save_dir: str = None,
                  filename: str = 'metrics') -> list:
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


def get_model_predictions(model: Model, test_generator: PlanGeneratorMultiPerc) -> list:
    y_pred = list()
    y_true = list()
    for i in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(i)
        y_pred.extend(model.predict(x))
        y_true.extend(y)
    return y_pred, y_true


def create_model_dir_name(params: dict, epochs: int, max_plan_percentage: float, batch_size: int):
    hidden_layer_dim = params['recurrent_list'][1]['units']
    embedding_dim = params['embedding_params']['output_dim']
    dropout = params['recurrent_list'][1]['dropout']
    recurrent_dropout = params['recurrent_list'][1]['recurrent_dropout']
    model_name = params['model_name']
    recurrent_type = params['recurrent_list'][0]
    loss_function = params['loss_function']

    model_dir_name = (f'{model_name}_{recurrent_type}_epochs={epochs}_embedding={embedding_dim}_units=' +
                      f'{hidden_layer_dim}_dropout={dropout}_recurrent-dropout={recurrent_dropout}_loss=' +
                      f'{loss_function}_plan-percentage={max_plan_percentage}_batch-size={batch_size}')

    return model_dir_name


def get_callback_default_params(callback_name: str) -> dict:
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


def objective(trial: optuna.Trial,
              model_name: str,
              train_plans: list,
              val_plans: list,
              dizionario: dict,
              dizionario_goal: dict,
              max_plan_dim: int,
              plan_percentage_min: float,
              plan_percentage_max: float,
              epochs: int,
              batch_size: int):
    if train_plans is not None and dizionario is not None and dizionario_goal is not None:
        train_generator = PlanGeneratorMultiPerc(train_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
                                                 plan_percentage_min, plan_percentage_max)
        val_generator = PlanGeneratorMultiPerc(val_plans, dizionario, dizionario_goal, batch_size, max_plan_dim,
                                               plan_percentage_min, plan_percentage_max, shuffle=False)

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

    use_activity_regularisation = False
    if use_activity_regularisation:
        l1 = trial.suggest_categorical('l1', [0.01, 0.001, 0.0001, 0.00001])
        l2 = trial.suggest_categorical('l2', [0.01, 0.001, 0.0001, 0.00001])
    else:
        l1 = None
        l2 = None

    params = ParamsGenerator(model_name=model_name,
                             recurrent_type=trial.suggest_categorical('recurrent_type', ['lstm', 'bilstm']),
                             units=trial.suggest_int('hidden_layer_dim', 150, 512),
                             output_dim=trial.suggest_int('embedding_dim', 50, 200),
                             dropout=dropout,
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


def run_tests(model: Model, test_plans: list, dizionario: dict, dizionario_goal: dict, batch_size: int,
              max_plan_dim: int,
              min_plan_perc: float, plan_percentage: float, save_dir: str, filename='metrics') -> None:
    if test_plans is not None:
        test_plans = test_plans[0]
        test_generator = PlanGeneratorMultiPerc(test_plans, dizionario, dizionario_goal, batch_size,
                                                max_plan_dim, min_plan_perc, plan_percentage, shuffle=False)
        y_pred, y_true = get_model_predictions(model, test_generator)
        scores = print_metrics(y_true=y_true, y_pred=y_pred, dizionario_goal=dizionario_goal, save_dir=save_dir,
                               filename=filename)


def create_study(study_name: str, db_dir: str) -> optuna.Study:
    study = optuna.create_study(
                storage=f'sqlite:///{join(db_dir, f"{study_name}.db")}',
                sampler=TPESampler(seed=43),
                direction='maximize',
                load_if_exists=True,
                study_name=study_name
            )
    return study


@click.group()
def run():
    pass

@run.group('train-model')
@click.pass_context
@click.option('--target-dir', 'target_dir', required=True, prompt=True, type=click.STRING,
              help=f'{HELPS.MODEL_DIR_OUT} {HELPS.CREATE_IF_NOT_EXISTS}')
@click.option('--plan-perc', 'max_plan_percentage', required=True, prompt=True, help=HELPS.MAX_PLAN_PERCENTAGE,
              type=click.FloatRange(0, 1))
@click.option('--batch-size', 'batch_size', default=64, type=click.INT, help=HELPS.BATCH_SIZE, show_default=True)
@click.option('--read-dict-dir', 'read_dict_dir', required=True, prompt=True,
              help=HELPS.DICT_FOLDER_SRC, type=click.STRING)
@click.option('--epochs', default=50, help=HELPS.EPOCHS, type=click.INT, show_default=True)
@click.option('--min-plan-perc', 'min_plan_percentage', default=0.3, type=click.FloatRange(0, 1),
              help=HELPS.MIN_PLAN_PERCENTAGE, show_default=True)
@click.option('--read-plans-dir', 'read_plans_dir', required=True, prompt=True,
              help=HELPS.PLANS_FOLDER_SRC, type=click.STRING)
@click.option('--max-plan-dim', 'max_plan_dim', required=True, prompt=True, help=HELPS.MAX_PLAN_LENGTH, type=click.INT)
def train_model(ctx, target_dir, max_plan_percentage, batch_size, read_dict_dir, epochs, min_plan_percentage,
        read_plans_dir, max_plan_dim):
    [dizionario, dizionario_goal] = load_from_pickles(read_dict_dir, ['dizionario', 'dizionario_goal'])

    max_plan_dim = int(max_plan_percentage * max_plan_dim)

    if dizionario is not None and dizionario_goal is not None and max_plan_dim > 0:
        ctx.ensure_object(dict)
        ctx.obj[KEYS.ACTION_DICT] = dizionario
        ctx.obj[KEYS.GOALS_DICT] = dizionario_goal
        ctx.obj[KEYS.EPOCHS] = epochs
        ctx.obj[KEYS.BATCH_SIZE] = batch_size
        ctx.obj[KEYS.MAX_PLAN_PERC] = max_plan_percentage
        ctx.obj[KEYS.MIN_PLAN_PERC] = min_plan_percentage
        ctx.obj[KEYS.READ_PLANS_DIR] = read_plans_dir
        ctx.obj[KEYS.MAX_PLAN_DIM] = max_plan_dim
        ctx.obj[KEYS.TARGET_DIR] = target_dir


@train_model.group('neural-network')
@click.pass_context
@click.option('--network-params', 'params_dir', required=True, prompt=True, help=HELPS.NETWORK_PARAMETERS_SRC,
              type=click.STRING)
def neural_network(ctx, params_dir):
    if params_dir is not None and ctx.ensure_object(dict):
        target_dir = ctx.obj[KEYS.TARGET_DIR]
        epochs = ctx.obj[KEYS.EPOCHS]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]

        params = load_file(params_dir,
                           load_ok=ERRORS.STD_LOAD_FILE_OK.format(os.path.basename(params_dir),
                                                                  os.path.dirname(params_dir)),
                           error=ERRORS.STD_ERROR_LOAD_FILE.format(os.path.basename(params_dir)))
        model_dir_name = create_model_dir_name(params, epochs, max_plan_percentage, batch_size)
        model_dir = join(target_dir, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)

        ctx.obj[KEYS.PARAMS] = params
        ctx.obj[KEYS.MODEL_DIR] = model_dir


@neural_network.command('train')
@click.pass_context
def network_train(ctx):
    if ctx.ensure_object(dict):
        params = ctx.obj[KEYS.PARAMS]
        model_dir = ctx.obj[KEYS.MODEL_DIR]
        dizionario = ctx.obj[KEYS.ACTION_DICT]
        dizionario_goal = ctx.obj[KEYS.GOALS_DICT]
        epochs = ctx.obj[KEYS.EPOCHS]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]
        min_plan_percentage = ctx.obj[KEYS.MIN_PLAN_PERC]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]

        model_name = params['model_name']
        plot_dir = join(model_dir, FILENAMES.NETWORK_PLOTS_FOLDER)
        os.makedirs(plot_dir, exist_ok=True)

        [train_plans, val_plans] = load_from_pickles(read_plans_dir, [FILENAMES.TRAIN_PLANS_FILENAME,
                                                                      FILENAMES.VALIDATION_PLANS_FILENAME])
        if train_plans is not None and dizionario is not None and dizionario_goal is not None:
            train_generator = PlanGeneratorMultiPerc(train_plans, dizionario, dizionario_goal,
                                                     batch_size, max_plan_dim, min_plan_percentage, max_plan_percentage)
            val_generator = None
            if val_plans is not None:
                val_generator = PlanGeneratorMultiPerc(val_plans, dizionario, dizionario_goal, batch_size,
                                                       max_plan_dim, min_plan_percentage, max_plan_percentage,
                                                       shuffle=False)

            model = build_network_single_fact(train_generator, **params)
            print_network_details(model, params)

            callbacks = [
                EarlyStopping(**get_callback_default_params('early_stopping')),
                ModelCheckpoint(filepath=join(model_dir, 'checkpoint'),
                                **get_callback_default_params('model_checkpoint'))
            ]
            history = train_network(model=model,
                                    train_generator=train_generator,
                                    callbacks=callbacks,
                                    validation_generator=val_generator,
                                    epochs=epochs)
            model.save(join(model_dir, f'{model_name}.h5'))
            save_plot(history=history, plot_dir=join(plot_dir, 'loss_plot.png'))
    else:
        print(ERRORS.MSG_ERROR_LOAD_PARAMS)


@neural_network.command('results')
@click.pass_context
@click.option('--incremental-tests', 'incremental_tests', is_flag=True, default=False,
              help=HELPS.INCREMENTAL_TESTS_FLAG)
def network_results(ctx, incremental_tests):
    if ctx.ensure_object(dict):
        params = ctx.obj[KEYS.PARAMS]
        model_dir = ctx.obj[KEYS.MODEL_DIR]
        dizionario = ctx.obj[KEYS.ACTION_DICT]
        dizionario_goal = ctx.obj[KEYS.GOALS_DICT]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]
        min_plan_percentage = ctx.obj[KEYS.MIN_PLAN_PERC]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]

        model_name = params['model_name']
        model = load_model(join(model_dir, f'{model_name}.h5'),
                           custom_objects={'AttentionWeights': AttentionWeights, 'ContextVector': ContextVector})
        print(model.summary())

        if incremental_tests:
            test_dir = join(read_plans_dir, 'incremental_test_sets')
            files = os.listdir(test_dir)
        else:
            test_dir = read_plans_dir
            files = [FILENAMES.TEST_PLANS_FILENAME]

        for f in files:
            test_plans = load_from_pickles(test_dir, [f])
            if not (test_plans is None) and len(test_plans) > 0:
                run_tests(model=model, test_plans=test_plans, dizionario=dizionario, dizionario_goal=dizionario_goal,
                          batch_size=batch_size, max_plan_dim=max_plan_dim, min_plan_perc=min_plan_percentage,
                          plan_percentage=max_plan_percentage, save_dir=model_dir, filename=f'metrics_{f}')
            else:
                print(f'Problems with file {f} in folder {test_dir}')


@train_model.command('optuna')
@click.pass_context
@click.option('--model-name', 'model_name', type=click.STRING, required=True, prompt=True,
              help=HELPS.MODEL_NAME)
@click.option('--trials', 'n_trials', default=20, type=click.INT, help=HELPS.TRIALS, show_default=True)
@click.option('--db-dir', 'db_dir', type=click.STRING, required=True, prompt=True, help=HELPS.DB_DIR)
def optuna_train(ctx, model_name, db_dir, n_trials):

    if ctx.ensure_object(dict):
        dizionario = ctx.obj[KEYS.ACTION_DICT]
        dizionario_goal = ctx.obj[KEYS.GOALS_DICT]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]
        min_plan_percentage = ctx.obj[KEYS.MIN_PLAN_PERC]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]
        epochs = ctx.obj[KEYS.EPOCHS]
        target_dir = ctx.obj[KEYS.TARGET_DIR]


        os.makedirs(db_dir, exist_ok=True)
        study = create_study(model_name, db_dir)

        ctx.ensure_object(dict)
        ctx.obj[KEYS.STUDY] = study
        ctx.obj[KEYS.MODEL_NAME] = model_name

        [train_plans, val_plans] = load_from_pickles(read_plans_dir, [FILENAMES.TRAIN_PLANS_FILENAME,
                                                                      FILENAMES.VALIDATION_PLANS_FILENAME])
        study.optimize(
            lambda trial: objective(trial=trial,
                                    dizionario=dizionario,
                                    dizionario_goal=dizionario_goal,
                                    model_name=model_name,
                                    train_plans=train_plans,
                                    val_plans=val_plans,
                                    max_plan_dim=max_plan_dim,
                                    plan_percentage_min=min_plan_percentage,
                                    plan_percentage_max=max_plan_percentage,
                                    epochs=epochs,
                                    batch_size=batch_size),
            n_trials=n_trials,
            gc_after_trial=True)

        plot_dir = join(target_dir, model_name)
        plot_dir = join(plot_dir, FILENAMES.NETWORK_PLOTS_FOLDER)
        os.makedirs(plot_dir, exist_ok=True)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(join(plot_dir, 'opt_hist.png'))

        fig = optuna.visualization.plot_slice(study)
        fig.write_image(join(plot_dir, 'slice.png'))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(join(plot_dir, 'param_importance.png'))


@run.command('results')
@click.option('--model-name', 'model_name', type=click.STRING, required=True, prompt=True,
              help=HELPS.MODEL_NAME)
@click.option('--db-dir', 'db_dir', type=click.STRING, required=True, prompt=True, help=HELPS.DB_DIR)
def optuna_results(model_name, db_dir):
        study = create_study(model_name, db_dir)
        print(study.best_params)


if __name__ == '__main__':
    np.random.seed(47)
    run()