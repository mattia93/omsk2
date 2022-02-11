import os


def create_folder_single_plan(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    command = f'cp test_files/logistics_test_plans/xml-LPG-p000001_2.SOL {target_dir}'
    os.system(command)

def delete_dir(target_dir: str) -> None:
    command = f'rm -r {target_dir}'
    os.system(command)
