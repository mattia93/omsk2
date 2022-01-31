import os
import numpy as np
import utils
import oneHot_deep
import click
from utils_functions import save_file
from constants import CREATE_DATASET, HELPS, FILENAMES


def create_dictionary(plans : list, oneHot : bool = True):
    dictionary = oneHot_deep.create_dictionary(plans)
    dictionary = oneHot_deep.shuffle_dictionary(dictionary)
    if oneHot:
      oneHot_deep.completa_dizionario(dictionary)
    return dictionary


def create_dictionary_goals_fixed(plans):
    goals = []
    for p in plans:
        if p.goals not in goals:
            goals.append(p.goals)
    print(CREATE_DATASET.GOALS_NUMBER.format(len(goals)))
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


def create_dictionary_goals_not_fixed(plans):
    goals = []
    for p in plans:
        for fact in p.goals:
            if fact not in goals:
                goals.append(fact)
    print(CREATE_DATASET.GOALS_NUMBER.format(len(goals)))
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


@click.command()
@click.option('--read-dir', 'read_dir', prompt=True, required=True, type=click.STRING,
              help=HELPS.XML_FOLDER_SRC)
@click.option('--target-dir', 'target_dir', prompt=True, required=True,
              type=click.STRING, help=f'{HELPS.PLANS_AND_DICT_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}')
@click.option('--onehot', is_flag=True, default=False, help=HELPS.ONEHOT_FLAG)
def run(read_dir, target_dir, onehot):

    os.makedirs(target_dir, exist_ok=True)

    plans = utils.get_all_plans(read_dir)
    dizionario = create_dictionary(plans, onehot)
    dizionario_goal = create_dictionary_goals_not_fixed(plans)

    save_file(plans, target_dir, FILENAMES.PLANS_FILENAME)
    save_file(dizionario, target_dir, FILENAMES.ACTION_DICT_FILENAME)
    save_file(dizionario_goal, target_dir, FILENAMES.GOALS_DICT_FILENEME)

if __name__ == '__main__':
    np.random.seed(47)
    run()











