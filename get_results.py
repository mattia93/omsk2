import numpy as np
import pickle
import sys
import pandas as pd
from sklearn.metrics import classification_report
import getopt
from matplotlib import pyplot as plt
import os
from os.path import join

def uniqueness(goals : set):
    first = next(iter(goals))
    count = np.zeros((len(first), ))
    size = len(goals)
    for goal in goals:
        count += goal

    values = np.ones((len(first), ))*size
    result = np.zeros((len(values),))
    result = np.divide(values, count, where=count != 0)
    return result

def get_values_from_dict(file_path : str):
    try:
        with open(file_path, 'rb') as f:
            res_dict = pickle.load(f)
            print(res_dict.keys())
            try:
                y_pred_test = [ e > 0.5 for e in res_dict['y_test_pred']]
                #print( res_dict['y_test_pred'])
                y_true_test = res_dict['y_test_true']
            except KeyError:
                y_pred_test = []
                y_true_test = []
            try:
                y_true_train = res_dict['y_train_true']
                y_pred_train = (res_dict['y_train_pred'] > 0.5)
            except KeyError:
                y_true_train = []
                y_pred_train = []
            print('Predictions loaded')
            return [y_true_test, y_pred_test, y_true_train, y_pred_train]
    except FileNotFoundError:
        print(f'Error while loading {file_path}\n'
              f'Please check the -r option is correct.')
        return [None, None, None, None]



def get_count_dict(goal_set : set, goal_list : np.ndarray):
    j = 0
    perc = 0
    count_dict = dict()
    for goal in goal_list:
        t_goal = tuple(goal)
        if t_goal in goal_set:
            try:
                count_dict[t_goal] = count_dict[t_goal] + 1
            except KeyError:
                count_dict[t_goal] = 1
        #if j >= perc*(len(goal_list)-2):
        #    sys.stdout.write('\r' + str(int(perc*100)) + '%')
        #    perc += 0.01
        #    sys.stdout.flush()
        #j +=1
    print('')
    return count_dict

def get_count_df(count_dict : dict, max_goal : int):
    keys = list(count_dict.keys())
    for i in range(len(keys)):
        keys[i] = str(keys[i])
    values = list(count_dict.values())

    df = pd.DataFrame()

    df['goal'] = keys

    df['count'] = values
    df.index = df['goal']
    df = df.sort_values(by=['count'], axis=0, ascending=False).iloc[0:max_goal]
    index = list(range(df.shape[0]))
    df['index'] = index
    return df

def get_score_df(count_df : pd.DataFrame, y_true : np.ndarray, y_pred : np.ndarray,
                 uniqueness : np.ndarray):
    df_score = pd.DataFrame()
    for i in range(len(y_true)):
        try:
            ## if y_true is in count_df then we can use it otherwise we pass
            count_df.loc[str(tuple(y_true[i]))]['index']
            ##
            score = list()
            for goal in count_df['goal']:
                a_goal = np.array(eval(goal))
                goal_score = np.multiply(y_pred[i], uniqueness)
                goal_score = np.dot(goal_score, a_goal)
                score.append(goal_score)
            df_score[i] = score
        except KeyError:
            pass
    index = list()
    for goal in count_df['goal']:
        index.append(goal)
    df_score.index = index
    return df_score

def get_max(df : pd.DataFrame, column : str):
    to_ret = list()
    max = 0
    for i, row in df.iterrows():
        if row[column] > max:
            max = row[column]
            to_ret = [i]
        elif row[column] == max:
            to_ret.append(i)
    return max, to_ret

def get_categorized_df(score_df : pd.DataFrame, count_df : pd.DataFrame, y_true : np.ndarray):
    y_true_cat = list()
    y_pred_cat = list()
    index = list()
    for c in list(score_df.columns):
        goal_true = str(tuple(y_true[int(c)]))
        #goal_pred = score_df[c].idxmax()
        m, max_list = get_max(score_df, c)
        goal_pred = max_list[np.random.randint(0, len(max_list))]
        y_true_cat.append(count_df['index'].loc[goal_true])
        y_pred_cat.append(count_df['index'].loc[goal_pred])
        index.append(int(c))
    cat_df = pd.DataFrame()
    cat_df['y_true'] = y_true_cat
    cat_df['y_pred'] = y_pred_cat
    cat_df['index'] = index

    return cat_df

def get_error_analysis(cat_df : pd.DataFrame, score_df : pd.DataFrame, count_df : pd.DataFrame):
    for i, row in cat_df.iterrows():
        if row['y_true'] != row['y_pred']:
            print(f"Actual sub-goal : {row['y_true']}, Predicted sub-goal : {row['y_pred']}")
            for g in list(score_df.index):
                print(f"{int(count_df['index'].loc[g])} : {float(score_df[row['index']].loc[g]):.2f}")


def show_plot(score_df : pd.DataFrame, print_count : bool = False, save_plot : bool = False, save_dict : str = './'):
    count_draws = 0
    count_single = 0
    count_zeros = 0
    col = []
    for c in score_df.columns:
        max, res = get_max(score_df, c)
        res = len(res)
        col.append(res)
        if print_count:
            if max == 0:
                count_zeros += 1
            elif res == 1:
                count_single += 1
            elif res > 1:
                count_draws += 1
            elif res < 1:
                print('ERROR')

    if print_count:
        print(f'{count_zeros = }\n{count_single = }\n{count_draws = }')

    df1 = pd.DataFrame()
    df1['max'] = col
    ax = df1.plot.hist(bins=range(1, 12))
    #plt.show()

def print_goals(count_df : pd.DataFrame, dizionario_goal : dict):
        for g_num, g1 in enumerate(count_df.index):
            print(f'GOAL {g_num}')
            for i, b in enumerate(eval(g1)):
                if b == 1:
                    for k in dizionario_goal:
                        if dizionario_goal[k][i] == 1:
                            print(f'({k})')
                            break

def get_uniqueness_goals(count_df : pd.DataFrame):
    uniq_goals = list()
    for goal in count_df['goal']:
        a_goal = np.array(eval(goal))
        goal_score = np.multiply(a_goal, uniq)
        uniq_goals.append(goal_score)
    return uniq_goals

if __name__ == '__main__':
    np.random.seed(47)
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'r:n:')
    read_folder = '.'
    output_folder = '.'
    filename = './dataset_embedding/gr6_retrain'#'./dataset_embedding/final_res_dict'#'final_res_renamed_dict'#'final_res_dict'#'results_new_not_fixed'#'first_test/final_res_dict'

    for opt, arg in opts:
        if opt == '-r':
            read_folder = arg
        elif opt == '-n':
             filename = arg

    filenames = os.listdir(read_folder)
    for filename in filenames:
        print(filename)
        [y_true_test, y_pred_test, y_true_train, y_pred_train] = get_values_from_dict(join(read_folder, filename))
        if y_true_test is None and y_true_train is None:
            print('Could not compute the results.')
        else:

            if len(y_true_test) > 0 and len(y_true_train)> 0:
                y_true_test = np.concatenate((y_true_train, y_true_test), axis=0)
                y_pred_test = np.concatenate((y_pred_train, y_pred_test), axis=0)
            elif len(y_true_test) == 0:
                y_pred_test = y_pred_train
                y_true_test = y_true_train
            goal_set = {tuple(row) for row in y_true_test}
            count_dict = get_count_dict(goal_set, y_true_test)
            count_df = get_count_df(count_dict, 10)
            sub_goal_set = set()
            for el in count_df.index:
                sub_goal_set.add(eval(el))
            uniq = uniqueness(sub_goal_set)
            score_df = get_score_df(count_df, y_true_test, y_pred_test, uniq)
            cat_df = get_categorized_df(score_df, count_df, y_true_test)
            goal_uniqueness = get_uniqueness_goals(count_df)
            show_plot(score_df, print_count=True)
            b = classification_report(cat_df['y_true'], cat_df['y_pred']).split('\n')
            b = [row for row in b if row != '']
            for i, row in enumerate(b):
                if i>0 and i<len(goal_uniqueness)+1:
                    print(row + (f'     {np.sum(goal_uniqueness[i-1]) : 4.2f}'))
                elif i == 0:
                    print(row + ('   Goal Uniqueness'))
                else:
                    print(row)
            #get_error_analysis(cat_df, score_df, count_df)
            #with open('./dataset_embedding/dizionario_goal', 'rb') as d:
            #    dizionario_goal = pickle.load(d)
            #    print_goals(count_df, dizionario_goal)
