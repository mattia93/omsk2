import plan
import os
from pathlib import Path
import pickle


def get_plans(folder):
    plan_list = [plan.Plan(folder + "/" + file) for file in os.listdir(folder) if file.find("xml") >= 0 and (file.endswith(".soln") or file.endswith(".SOL"))]
    return plan_list


def get_plans(folder, max_actions):
    plan_list = [plan.Plan(folder + "/" + file) for file in os.listdir(folder) if file.find("xml") >= 0 and (file.endswith(".soln") or file.endswith(".SOL") or file.endswith(".sol"))]
    #plan_list_max_actions = [plan for plan in plan_list if len(plan.actions) <= max_actions]
    #return plan_list_max_actions
    return plan_list


def get_folders(argv, name):
    if len(argv) < 2:
        raise ValueError('Please specify the input folder')
    read_folder = argv[1]
    if len(argv) == 3:
        save_folder = argv[2]
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        save_path = str(Path(save_folder)/name)
    else:
        save_path = Path("./")/name
    return read_folder, str(save_path)
    
def load_files(read_file : str, load_ok : str = 'File loaded', error : str = f'Error while loading file') -> list:
    try:
        with open(read_file, 'rb') as rf:
            plans = pickle.load(rf)
            print(load_ok)
    except FileNotFoundError:
        print(error)
        plans = None

    return plans