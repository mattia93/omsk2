import os
from os.path import join
import click
from constants import EXPERIMENT, HELPS


@click.command()
@click.option(
    "--python",
    "python_path",
    type=click.STRING,
    default="python",
    show_default=True,
    help=HELPS.PYTHON_FILE,
)
@click.option(
    "--root-dir",
    "root_dir",
    type=click.STRING,
    required=True,
    help=HELPS.STARTING_DIR_SRC,
)
@click.option(
    "--domain",
    "domain",
    type=click.STRING,
    required=True,
    help=HELPS.DOMAIN,
    prompt=True,
)
@click.option(
    "--dataset-type",
    "dataset_type",
    type=click.STRING,
    default="tasks_simil-pereira",
    show_default=True,
    help=HELPS.DATASET_TYPE,
)
@click.option(
    "--solver-path",
    "solver_path",
    type=click.STRING,
    help=HELPS.SOLVER_FILE_SRC,
    prompt=True,
    required=True,
)
@click.option(
    "--solutions",
    "solutions",
    type=click.INT,
    help=HELPS.SOLUTIONS_NUMBER,
    default=4,
    show_default=True,
)
@click.option(
    "--cpu-time",
    "cpu_time",
    type=click.INT,
    help=HELPS.CPU_TIME,
    default=30,
    show_default=True,
)
@click.option(
    "--processors",
    "processors",
    type=click.INT,
    help=HELPS.PROCESSORS_NUMBER,
    default=1,
    show_default=True,
)
def gen_dataset(
    python_path,
    root_dir,
    domain,
    dataset_type,
    solver_path,
    solutions,
    cpu_time,
    processors,
):
    task_folder = join(root_dir, EXPERIMENT.DATASET_FOLDER, domain, dataset_type)
    plans_dir = join(task_folder, EXPERIMENT.PLANS_FOLDER)
    solutions_dir = join(task_folder, EXPERIMENT.SOLUTIONS_FOLDER)
    xmls_dir = join(task_folder, EXPERIMENT.XMLS_FOLDERS)
    plans_and_dict_dir = join(task_folder, EXPERIMENT.PLANS_AND_DICT_FOLDER)
    plots_dir = join(task_folder, EXPERIMENT.PLOTS_FOLDER)
    generate_sol_command = (
        f"{python_path} {EXPERIMENT.GENERATE_FILES_PY} sol --read-dir {plans_dir} --solver-path {solver_path} "
        f"--target-dir {solutions_dir} --solutions {solutions} --cpu-time {cpu_time} --processors {processors}"
    )
    generate_xml_command = (
        f"{python_path} {EXPERIMENT.GENERATE_FILES_PY} xml --read-dir {plans_dir} --solver-path {solver_path} "
        f"--target-dir {xmls_dir} --solutions-dir {solutions_dir} --solutions {solutions} --processors {processors} "
        f"--solutions-solver LPG"
    )
    generate_dataset_command = (
        f"{python_path} {EXPERIMENT.CREATE_DATASET_PY} --read-dir {xmls_dir} --target-dir {plans_and_dict_dir} "
        f"--plots-dir {plots_dir} --save-stats"
    )

    os.system(generate_sol_command)
    os.system(generate_xml_command)
    os.system(generate_dataset_command)


if __name__ == "__main__":
    gen_dataset()
