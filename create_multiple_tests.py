import pickle
from os.path import join, dirname, basename
from utils_functions import load_files, create_table, create_plot
import numpy as np
import random
import  matplotlib.pyplot as plt
import getopt
import sys
import os


def print_plans_stat(plans: list, nbins: int = 10, save_graph: str = None) -> list:
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
    return plans_len






if __name__ == '__main__':

    plot_dir = './'
    std_error_message = 'Error while loading {0}'
    std_ok_message = '{0} loaded from {1}'
    read_dir = './dataset_whole2_small'
    file_names = ['test_plans']
    target_dir = 'dataset_whole2_small'
    compute_stats = False
    save_test_sets = False
    create_validation = True
    max_plan_dim = 100
    train_percentage = 0.8
    intervals_number = 5

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', ['read-dir=', 'plot-dir=', 'target-dir=', 'stats',
                                          'max-plan-dim=', 'train-percentage=',
                                          'no-validation', 'save-sets', 'intervals='])
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
        elif opt == '--save-sets':
            save_test_sets = True
        elif opt == '--no-val':
            create_validation = False
        elif opt == '--intervals':
            intervals_number = int(arg)


    plans = list()
    for file_name in file_names:
        f = join(read_dir, file_name)
        p = load_files(f,
                       error=std_error_message.format(f),
                       load_ok=std_ok_message.format(basename(f).capitalize(), dirname(f)))
        if p is not None:
            plans.extend(p)
    if len(plans) > 0 and compute_stats:
        os.makedirs(plot_dir, exist_ok=True)
        plans_len = print_plans_stat(plans, nbins=30)
        intervals = list(np.linspace(min(plans_len), max(plans_len), intervals_number))
        row = np.zeros((intervals_number-1,))
        headers = list()
        test_sets = list()
        for i in range(len(intervals)-1):
            headers.append(f'{int(intervals[i])}-{int(intervals[i+1])}')
            test_sets.append(list())
        for p in plans:
            for i in range(len(intervals)-1):
                if intervals[i] <= len(p.actions) < intervals[i+1]:
                    row[i] += 1
                    test_sets[i].append(p)
                    break

        table = create_table('Intervals dimensions', headers, [row])
        for row in table:
            print(row)

    if len(plans) > 0 and save_test_sets:
        set_dim = 1000
        random.seed(43)
        os.makedirs(target_dir, exist_ok=True)
        for i, ts in enumerate(test_sets):
            if len(ts) >= set_dim:
                ts = list(random.sample(ts, set_dim))
                with open(join(target_dir, f'test_plans_{headers[i]}'), 'wb') as f:
                    pickle.dump(ts, f)
            else:
                print(f'Element {headers[i]} has less than {set_dim} elements ({len(ts)})')

#        os.makedirs(target_dir, exist_ok=True)
#        with open(join(target_dir, 'train_plans'), 'wb') as f:
#            pickle.dump(train_plans, f)
#
#        if len(val_plans) > 0:
#            with open(join(target_dir, 'val_plans'), 'wb') as f:
#                pickle.dump(val_plans, f)

#        if len(test_plans) > 0:
#            with open(join(target_dir, 'test_plans'), 'wb') as f:
#                pickle.dump(test_plans, f)





