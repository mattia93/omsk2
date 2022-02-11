import utils
from tests.tests_utils import create_folder_single_plan, delete_dir
from create_dataset import create_dictionary, create_dictionary_goals_not_fixed


def test_create_dictionary():

    test_dicts_dir = "tests/test_dict_files"
    create_folder_single_plan(test_dicts_dir)
    key = "DRIVE-TRUCK TRU1 POS13 POS22 CIT3"

    plans = utils.get_all_plans(test_dicts_dir)
    d = create_dictionary(plans, False)

    s = set()
    for a in plans[0].actions:
        s.add(a.name)
    assert len(d) == len(s)
    assert key in d.keys()
    assert type(d[key]) == int and d[key] >= 0 and d[key] < len(d)

    d = create_dictionary(plans, True)
    assert key in d.keys()
    assert len(d[key] == len(d))
    delete_dir(test_dicts_dir)


def test_create_goals_dict():
    test_dicts_dir = "tests/test_goals_files"
    create_folder_single_plan(test_dicts_dir)

    plans = utils.get_all_plans(test_dicts_dir)
    d = create_dictionary_goals_not_fixed(plans)

    s = set()
    for g in plans[0].goals:
        print(g)
        s.add(g)

    assert len(d) == len(s)
    delete_dir(test_dicts_dir)
