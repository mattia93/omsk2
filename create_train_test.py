from os.path import join, dirname, basename
from utils_functions import load_file, create_table, create_plot, save_file
import numpy as np
import random
import click
import os
from constants import ERRORS, FILENAMES, CREATE_TRAIN_TEST, HELPS, KEYS




def print_plans_stat(plans: list, nbins: int = 10, save_graph: str = None) -> None:
    print(CREATE_TRAIN_TEST.PLANS_NUMBER.format(len(plans)))
    plans_len = list()
    for p in plans:
        plans_len.append(len(p.actions))

    headers = ['MIN', 'Q1', 'Q2', 'Q3', 'MAX']
    rows = list()
    row = list()
    for i in range(len(headers)):
        row.append(np.quantile(plans_len, i/4))
    rows.append(row)
    table = create_table('Plans length', headers, rows)
    for row in table:
        print(row)

    create_plot(plot_type='hist', target_dir=save_graph, input=plans_len, nbins=nbins, )


def print_action_distrib(plans: list, save_graph: str = None, nbins: int =10) -> None:
    freq_action_dict = dict()
    for p in plans:
        for a in p.actions:
            a = a.name
            if a in freq_action_dict.keys():
                freq_action_dict[a] +=1
            else:
                freq_action_dict[a] = 1

    print(CREATE_TRAIN_TEST.ACTIONS_NUMBER.format(len(freq_action_dict)))
    v = list(freq_action_dict.values())
    headers = ['MIN', 'Q1', 'Q2', 'Q3', 'MAX']
    rows = list()
    row = list()
    for i in range(len(headers)):
        row.append(np.quantile(v, i / 4))
    rows.append(row)

    table = create_table('Actions frequency', headers, rows)
    for row in table:
        print(row)

    create_plot(plot_type='hist', input=v, nbins=nbins, target_dir=save_graph)


def print_goal_distrib(plans: list, save_graph: str = None, nbins: int = 10):
    goals_dict = dict()
    for p in plans:
        for g in p.goals:
            if g in goals_dict.keys():
                goals_dict[g] = goals_dict[g] + 1
            else:
                goals_dict[g] = 1

    print(CREATE_TRAIN_TEST.GOALS_NUMBER.format(len(goals_dict)))
    v = list(goals_dict.values())
    headers = ['MIN', 'Q1', 'Q2', 'Q3', 'MAX']
    rows = list()
    row = list()
    for i in range(len(headers)):
        row.append(np.quantile(v, i / 4))
    rows.append(row)

    table = create_table('Goals frequency', headers, rows)
    for row in table:
        print(row)

    create_plot(plot_type='hist', input=v, nbins=nbins, target_dir=save_graph)



@click.group()
@click.pass_context
@click.option('--read-dir', 'read_dir', type=click.STRING, prompt=True, required=True,
               help=HELPS.PLANS_AND_DICT_FOLDER_SRC)
def cli(ctx, read_dir):
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
        ctx.ensure_object(dict)
        ctx.obj[KEYS.PLANS] = plans

@cli.command('stats')
@click.option('--target-dir', 'target_dir', prompt=True, required=True,
              type=click.STRING, help=f'{HELPS.PLOTS_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}')
@click.pass_context
def stats(ctx, target_dir):
    if ctx.ensure_object(dict):
        plans = ctx.obj[KEYS.PLANS]
        os.makedirs(target_dir, exist_ok=True)
        print_plans_stat(plans, nbins=30, save_graph=join(target_dir, FILENAMES.PLOT_LENGTH_FILENAME))
        print_action_distrib(plans, nbins=30, save_graph=join(target_dir, FILENAMES.PLOT_ACTIONS_FILENAME))
        print_goal_distrib(plans, nbins=30, save_graph=join(target_dir, FILENAMES.PLOT_GOALS_FILENAME))
    else:
        print(ERRORS.MSG_ERROR_LOAD_PLANS)

@cli.command('train-split')
@click.pass_context
@click.option('--target-dir', 'target_dir', prompt=True, required=True, type=click.STRING,
              help=f'{HELPS.TRAIN_TEST_VAL_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}')
@click.option('--max-plan-dim', 'max_plan_dim', prompt=True, required=True, type=click.INT,
              help=HELPS.MAX_PLAN_LENGTH)
@click.option('--train-perc', 'train_percentage', default=0.8, type=click.FloatRange(0, 1),
              help=HELPS.TRAIN_PERCENTAGE)
@click.option('--no-val', 'create_validation', is_flag=True, default=True,
              help=HELPS.NO_VAL_FLAG, flag_value=False)
def train_split(ctx, target_dir, max_plan_dim, train_percentage, create_validation):
    if ctx.ensure_object(dict):
        random.seed(43)
        plans = ctx.obj[KEYS.PLANS]

        plans = [p for p in plans if len(p.actions) <= max_plan_dim]
        random.shuffle(plans)
        train_dim = int(train_percentage * len(plans))
        print(CREATE_TRAIN_TEST.TRAIN_PLANS_NUMBER.format(train_dim))
        if create_validation:
            val_dim = int((1-train_percentage)/2 * len(plans))
            print(CREATE_TRAIN_TEST.VALIDATION_PLANS_NUMBER.format(val_dim))
        else:
            val_dim = 0
        print(CREATE_TRAIN_TEST.TEST_PLANS_NUMBER.format(len(plans)-val_dim-train_dim))

        train_plans = plans[:train_dim]
        val_plans = plans[train_dim:val_dim+train_dim]
        test_plans = plans[val_dim+train_dim:]

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
    cli()





