import os
import numpy as np
import utils
import save_arrays
import oneHot_deep
import click


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
    print("ci sono:"+str(len(goals))+" goal")
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
    print("ci sono:"+str(len(goals))+" goal")
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


@click.command()
@click.option('--read-dir', 'read_dir', prompt=True, required=True, type=click.STRING,
              help='Folder that contains the XMLs files.')
@click.option('--target-dir', 'target_dir', prompt=True, required=True,
              type=click.STRING, help=("Folder where to store plans file. It's created if it does not exists."))
@click.option('--onehot', is_flag=True, default=False, help=('Flag that applies the one-hot representation for the '+
                                                             'goals.'))
def run(read_dir, target_dir, onehot):

    os.makedirs(target_dir, exist_ok=True)

    plans = utils.get_all_plans(read_dir)
    dizionario = create_dictionary(plans, onehot)
    dizionario_goal = create_dictionary_goals_not_fixed(plans)

    save_arrays.save(plans, os.path.join(target_dir, 'plans'))
    save_arrays.save(dizionario, os.path.join(target_dir, 'dizionario'))
    save_arrays.save(dizionario_goal, os.path.join(target_dir, 'dizionario_goal'))

if __name__ == '__main__':
    np.random.seed(47)
    run()











