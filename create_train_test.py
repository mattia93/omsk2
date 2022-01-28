import pickle
from os.path import join, dirname, basename
from utils_functions import load_file, create_table, create_plot
import numpy as np
import random
import  matplotlib.pyplot as plt
import getopt
import sys
import os


def print_plans_stat(plans: list, nbins: int = 10, save_graph: str = None) -> None:
    print(f'Total plans : {len(plans)}')
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

    print(f'There are {len(freq_action_dict)} actions')
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

    print(f'There are {len(goals_dict)} goals')
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






if __name__ == '__main__':

    plot_dir = './'
    std_error_message = 'Error while loading {0}'
    std_ok_message = '{0} loaded from {1}'
    read_dir = './dataset_whole2_small'
    file_names = ['plans']
    target_dir = 'dataset_whole2_small'
    compute_stats = False
    split_train_test = False
    create_validation = True
    max_plan_dim = 100
    train_percentage = 0.8

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', ['read-dir=', 'plot-dir=', 'target-dir=', 'stats',
                                          'max-plan-dim=', 'train-percentage=',
                                          'no-validation', 'train-split'])
    for opt, arg in opts:
        if opt == '--read-dir':
            read_dir = arg
        elif opt == '--plot-dir':
            plot_dir = arg
        elif opt == '--target-dir':
            target_dir = arg
        elif opt == '--stats':
            compute_stats = True
        elif opt == '--max-plan-dim':
            max_plan_dim = int(arg)
        elif opt == '--train-percentage':
            train_percentage = float(arg)
        elif opt == '--train-split':
            split_train_test = True
        elif opt == '--no-val':
            create_validation = False


    plans = list()
    for file_name in file_names:
        f = join(read_dir, file_name)
        p = load_file(f,
                       error=std_error_message.format(f),
                       load_ok=std_ok_message.format(basename(f).capitalize(), dirname(f)))
        if p is not None:
            plans.extend(p)
    if len(plans) > 0 and compute_stats:
        os.makedirs(plot_dir, exist_ok=True)
        print_plans_stat(plans, nbins=30, save_graph=join(plot_dir, 'plans_length.png'))
        print_action_distrib(plans, nbins=30, save_graph=join(plot_dir, 'action_frequency.png'))
        print_goal_distrib(plans, nbins=30, save_graph=join(plot_dir, 'goal_frequency.png'))

    if len(plans) > 0 and split_train_test:
        random.seed(43)
        plans = [p for p in plans if len(p.actions) <= max_plan_dim]
        random.shuffle(plans)
        train_dim = int(train_percentage * len(plans))
        if create_validation:
            val_dim = int((1-train_percentage)/2 * len(plans))
        else:
            val_dim = 0

        train_plans = plans[:train_dim]
        val_plans = plans[train_dim:val_dim+train_dim]
        test_plans = plans[val_dim+train_dim:]

        target_dir = join(target_dir, f'plans_max-plan-dim={max_plan_dim}'
                                      f'_train_percentage={train_percentage}')
        os.makedirs(target_dir, exist_ok=True)
        with open(join(target_dir, 'train_plans'), 'wb') as f:
            pickle.dump(train_plans, f)

        if len(val_plans) > 0:
            with open(join(target_dir, 'val_plans'), 'wb') as f:
                pickle.dump(val_plans, f)

        if len(test_plans) > 0:
            with open(join(target_dir, 'test_plans'), 'wb') as f:
                pickle.dump(test_plans, f)





