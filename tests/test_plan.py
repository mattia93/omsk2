import os.path

from tests.tests_utils import create_folder_single_plan, delete_dir
from utils import get_all_plans


def test_plan():
    test_plans_dir = "tests/test_plans_file"
    create_folder_single_plan(test_plans_dir)
    plans = get_all_plans(test_plans_dir)
    assert len(plans) == 1
    assert len(plans[0].actions) == 37
    assert plans[0].actions[0].name == "DRIVE-TRUCK TRU1 POS13 POS22 CIT3"
    assert plans[0].actions[6].name == "LOAD-TRUCK OBJ66 TRU1 POS22"
    assert plans[0].plan_name == os.path.join(test_plans_dir, "xml-LPG-p000001_2.SOL")
    delete_dir(test_plans_dir)
