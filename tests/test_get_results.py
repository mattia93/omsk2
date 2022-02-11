from get_results import (
    get_used_output_list,
    get_representation,
    get_predictions,
    get_contributes,
    get_max,
)
import numpy as np


def test_get_used_output_list():

    y_true_list = [[1, 1, 1, 0], [0, 1, 1, 1]]

    used_output_list = get_used_output_list(y_true_list)
    assert len(used_output_list) == 4
    assert 0 in used_output_list
    assert 1 in used_output_list
    assert 2 in used_output_list
    assert 3 in used_output_list


def test_get_representation():
    y_true_list = [[1, 1, 1, 0], [0, 1, 1, 1]]
    used_output_list = [0, 1, 2, 3]

    goals = list()
    for y_true in y_true_list:
        representation = get_representation(y_true, used_output_list)
        goals.append(representation)
    goals_list_repr = list(set(goals))

    assert len(goals_list_repr) == 2
    assert "0, 1, 2" in goals_list_repr
    assert "1, 2, 3" in goals_list_repr


def test_get_predictions():
    y_true_list = [[1, 1, 1, 0], [0, 1, 1, 1]]
    y_pred_list = [[0.4, 0.7, 0.8, 0.1], [0.4, 0.7, 0.8, 0.1]]

    goals_list_repr = list()
    goals_list_repr.append("0, 1, 2")
    goals_list_repr.append("1, 2, 3")

    goals_list_bin = np.zeros((len(goals_list_repr), len(y_true_list[0])))
    for row, goal in enumerate(goals_list_repr):
        numbers = goal.split(",")
        for number in numbers:
            number = int(number.strip())
            goals_list_bin[row, number] = 1

    y_true, y_pred = get_predictions(
        y_pred_list=y_pred_list,
        y_true_list=y_true_list,
        goals_list_bin=goals_list_bin,
        use_threshold=False,
    )

    assert y_true == ["0, 1, 2", "1, 2, 3"]
    assert y_pred == ["0, 1, 2", "0, 1, 2"]


def test_get_contributes():
    y_true_list = [[1, 1, 1, 0], [0, 1, 1, 1]]
    y_pred_list = [[0.4, 0.8, 0.8, 0.1], [0.4, 0.8, 0.8, 0.1]]

    goals_list_repr = list()
    goals_list_repr.append("0, 1, 2")
    goals_list_repr.append("1, 2, 3")

    goals_list_bin = np.zeros((len(goals_list_repr), len(y_true_list[0])))
    for row, goal in enumerate(goals_list_repr):
        numbers = goal.split(",")
        for number in numbers:
            number = int(number.strip())
            goals_list_bin[row, number] = 1

    c = get_contributes(goals_list_bin, y_pred_list[0], y_true_list[0], y_true_list)
    assert c[0][0] == y_pred_list[0][0]
    assert c[0][1] == y_pred_list[0][1]
    assert c[0][2] == y_pred_list[0][2]
    assert c[0][3] == 0
    assert c[1][0] == 0
    assert c[1][1] == y_pred_list[1][1]
    assert c[1][2] == y_pred_list[1][2]
    assert c[1][3] == y_pred_list[1][3]


def test_get_max():
    contributes = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]]
    m = get_max(contributes)
    assert len(m) == 1
    assert m[0] == 2

    contributes = [[1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 1]]
    m = get_max(contributes)
    assert len(m) == 2
    assert m[0] == 0
    assert m[1] == 2

    contributes = []
    m = get_max(contributes)
    assert len(m) == 0
