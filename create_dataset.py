import os

import numpy as np
import utils
import sys
import save_arrays
import oneHot_deep
import getopt


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


if __name__ == '__main__':
    argv = sys.argv[1:]
    np.random.seed(47)
    opts, args = getopt.getopt(argv, "r:s:a:p:vh")
    read_folder, save_folder, max_actions, perc_dataset = 'xml_prova', 'dataset_prova', 100, 0.8
    use_validation = False
    use_onehot = False
    for opt, arg in opts:
        if opt == "-r":
            read_folder = arg
        elif opt == "-s":
            save_folder = arg
        elif opt == "-a":
            max_actions = int(arg)
        elif opt == "-p":
            perc_dataset = float(arg)
        elif opt == '-v':
            use_validation = True
        elif opt == '-h':
            use_onehot = True

    os.makedirs(save_folder, exist_ok=True)
            
    plans = utils.get_plans(read_folder, max_actions)
    dizionario = create_dictionary(plans, use_onehot)
    dizionario_goal = create_dictionary_goals_not_fixed(plans)

    save_arrays.save(plans, save_folder + '/plans')
    save_arrays.save(dizionario, save_folder + '/dizionario')
    save_arrays.save(dizionario_goal, save_folder + '/dizionario_goal')









