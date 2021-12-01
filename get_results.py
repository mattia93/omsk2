import os

from pandas import DataFrame
import pickle
import numpy as np
from sklearn.metrics import classification_report
import sys, getopt

def get_used_output_list(y_true_list: np.ndarray) -> np.ndarray:
    used_output = np.zeros((len(y_true_list[0]),), dtype=bool)
    for output in range(len(used_output)):
        for y_true in y_true_list:
            if y_true[output] == 1:
                used_output[output] = True
    used_output_index = np.array(np.where(used_output == True)).flatten()
    return used_output_index

def get_representation(y_true: np.ndarray, used_output_index: np.ndarray) -> str:
    representation = ''
    for index in used_output_index:
        if y_true[index]:
            representation += f'{index}, '
    return representation.strip()[:-1]

def get_uniqueness_values(goals_list_bin: np.ndarray) -> list:
    n_goals = len(goals_list_bin)
    uniq_values = np.sum(goals_list_bin, axis=0)
    uniq_values = [ n_goals/el if el != 0 else 0 for el in uniq_values]
    return uniq_values

def get_contributes(goals_list_bin: np.ndarray, pred: np.ndarray, y_true: np.ndarray, verbose: bool = False,
                    used_output_index: np.ndarray = None, uniq_values: np.ndarray = None) -> np.ndarray:
    contributes = None
    if used_output_index is None:
        used_output_index = get_used_output_list(y_true_list)
    if uniq_values is None:
        uniq_values = get_uniqueness_values(goals_list_bin)
    for i in range(len(goals_list_bin)):
        value = np.dot(np.multiply(pred, goals_list_bin[i]),np.transpose(uniq_values))
        if contributes is None:
            contributes = np.multiply(np.multiply(pred, goals_list_bin[i]), uniq_values)
        else:
            contributes = np.vstack([contributes, np.multiply(np.multiply(pred, goals_list_bin[i]),
                                                                              uniq_values)])
        if verbose:
            print([f'{j[0]}:{value:.5f}' for j, value in np.ndenumerate(np.multiply(pred, goals_list_bin[i])) if value > 0])
            print([f'{j[0]}:{value:.5f}' for j, value in np.ndenumerate(np.multiply(np.multiply(pred, goals_list_bin[i]),
                                                                                    uniq_values)) if value > 0])
            print(f'{get_representation(goals_list_bin[i], used_output_index)}: {value:.5f}')
    if verbose:
        print(f'Correct_goal: {get_representation(y_true, used_output_index)}')
    return contributes

def get_max(contribute: np.ndarray):
    max_element = -1
    index_max = list()
    for i in range(len(contribute)):
        if sum(contribute[i]) > max_element:
            max_element = sum(contribute[i])
            index_max = [i]
        elif sum(contribute[i]) == max_element:
            index_max.append(i)

    return index_max

def get_predictions(y_pred_list: np.ndarray, y_true_list: np.ndarray, goals_list_bin: np.ndarray,
                    use_threshold: bool = False) -> list:
    used_output_index = get_used_output_list(y_true_list)
    goal_preds = list()
    goal_true = list()
    uniq_values = get_uniqueness_values(goals_list_bin)
    single_count = 0
    draw_count = 0
    zeros_count = 0
    for i, y_pred in enumerate(y_pred_list):
        if use_threshold:
            y_pred = [1 if el > 0.5 else 0 for el in y_pred]
        y_true = y_true_list[i]
        contribute = get_contributes(goals_list_bin=goals_list_bin, pred=y_pred, y_true=y_true,
                                          used_output_index=used_output_index, uniq_values=uniq_values)

        index_max = get_max(contribute)


        if len(index_max) == 1:
            index = index_max[0]
            single_count +=1
        elif len(index_max) > 1:
            index = index_max[np.random.randint(0, len(index_max))]
            if len(index_max) == len(goals_list_bin):
                zeros_count +=1
            else:
                draw_count+=1
        else:
            print('ERROR')
            return None

        goal_preds.append(get_representation(goals_list_bin[index], used_output_index))
        goal_true.append(get_representation(y_true, used_output_index))
    print(f'{single_count = }\n{draw_count = }\n{zeros_count = }')
    return goal_true, goal_preds


if __name__ == '__main__':
    np.random.seed(47)
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'r:n:')
    read_folder = '.'
    output_folder = '.'
    filename = 'test/test_plans_p01.pkl'

    for opt, arg in opts:
        if opt == '-r':
            read_folder = arg
        elif opt == '-n':
             filename = arg

    if filename is None:
        files = os.listdir(read_folder)
    else:
        files = [filename]

    for file in files:
        with open(os.path.join(read_folder, file), 'rb') as rf:
            pred_dict = pickle.load(rf)
            y_pred_list = pred_dict['y_test_pred']
            y_true_list = pred_dict['y_test_true']

        used_output_index = get_used_output_list(y_true_list)
        goals = list()
        for y_true in y_true_list:
            representation = get_representation(y_true, used_output_index)
            goals.append(representation)
        goals_list_repr = list(set(goals))
        goals_list_bin = np.zeros((len(goals_list_repr), len(y_true_list[0])))
        for row, goal in enumerate(goals_list_repr):
            numbers = goal.split(',')
            for number in numbers:
                number = int(number.strip())
                goals_list_bin[row, number] = 1
        uniq_values = get_uniqueness_values(goals_list_bin=goals_list_bin)
        y_true, y_pred = get_predictions(y_pred_list=y_pred_list, y_true_list=y_true_list,
                                         goals_list_bin=goals_list_bin, use_threshold=False)
        print(classification_report(y_true=y_true, y_pred=y_pred))




