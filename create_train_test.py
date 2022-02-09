from os.path import join, dirname, basename
from utils_functions import load_file, save_file
import numpy as np
import random
import click
import os
from constants import ERRORS, FILENAMES, CREATE_TRAIN_TEST, HELPS, KEYS


@click.command()
@click.option('--read-dir', 'read_dir', type=click.STRING, prompt=True, required=True,
               help=HELPS.PLANS_AND_DICT_FOLDER_SRC)
@click.option('--target-dir', 'target_dir', prompt=True, required=True, type=click.STRING,
              help=f'{HELPS.TRAIN_TEST_VAL_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}')
@click.option('--max-plan-dim', 'max_plan_dim', prompt=True, required=True, type=click.INT,
              help=HELPS.MAX_PLAN_LENGTH)
@click.option('--train-perc', 'train_percentage', default=0.8, type=click.FloatRange(0, 1),
              help=HELPS.TRAIN_PERCENTAGE)
@click.option('--no-val', 'create_validation', is_flag=True, default=True,
              help=HELPS.NO_VAL_FLAG, flag_value=False)
def run(read_dir, target_dir, max_plan_dim, train_percentage, create_validation):
    file_names = [FILENAMES.PLANS_FILENAME]
    plans = list()
    for file_name in file_names:
        f = join(read_dir, file_name)
        p = load_file(f,
                      error=ERRORS.STD_ERROR_LOAD_FILE.format(f),
                      load_ok=ERRORS.STD_LOAD_FILE_OK.format(basename(f).capitalize(), dirname(f)))
        if p is not None:
            plans.extend(p)
    if len(plans) > 0:
        plans = [p for p in plans if len(p.actions) <= max_plan_dim]
        random.shuffle(plans)
        train_dim = int(train_percentage * len(plans))
        print(CREATE_TRAIN_TEST.TRAIN_PLANS_NUMBER.format(train_dim))
        if create_validation:
            val_dim = int((1 - train_percentage) / 2 * len(plans))
            print(CREATE_TRAIN_TEST.VALIDATION_PLANS_NUMBER.format(val_dim))
        else:
            val_dim = 0
        print(CREATE_TRAIN_TEST.TEST_PLANS_NUMBER.format(len(plans) - val_dim - train_dim))

        train_plans = plans[:train_dim]
        val_plans = plans[train_dim:val_dim + train_dim]
        test_plans = plans[val_dim + train_dim:]

        target_dir = join(target_dir, f'plans_max-plan-dim={max_plan_dim}'
                                      f'_train_percentage={train_percentage}')
        os.makedirs(target_dir, exist_ok=True)
        save_file(train_plans, target_dir, FILENAMES.TRAIN_PLANS_FILENAME)
        if len(val_plans) > 0:
            save_file(val_plans, target_dir, FILENAMES.VALIDATION_PLANS_FILENAME)
        else:
            print(ERRORS.STD_FILE_NOT_SAVED.format(FILENAMES.VALIDATION_PLANS_FILENAME))
        if len(test_plans) > 0:
            save_file(test_plans, target_dir, FILENAMES.TEST_PLANS_FILENAME)
        else:
            print(ERRORS.STD_FILE_NOT_SAVED.format(FILENAMES.TEST_PLANS_FILENAME))
    else:
        print(ERRORS.MSG_ERROR_LOAD_PLANS)


if __name__ == '__main__':
    run()





