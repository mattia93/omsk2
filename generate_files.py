import os
from multiprocessing import Pool
from os.path import isfile, join, basename, dirname
from functools import partial
import click
from constants import HELPS, GENERATE_FILES


def clean_solution_file(f: str) -> None:
    with open(f, "r") as fin:
        data = fin.read().splitlines(True)
    if len(data) > 0 and data[1].startswith(";"):
        with open(f, "w") as fout:
            fout.writelines(data[11:])


def get_basename_dirname(problem_path: str) -> list:
    problem_basename = basename(problem_path)
    problem_basename = problem_basename.split(".")[0]
    problem_dir = dirname(problem_path)
    return problem_dir, problem_basename


def execute_create_solution(
    problem_path: str,
    solver_path: str,
    output_dir: str,
    sol_number: int = 4,
    cpu_time: int = 20,
    solver: str = GENERATE_FILES.LPG,
):

    problem_dir, problem_basename = get_basename_dirname(problem_path)
    copy_solver_command = (
        f"cp {solver_path} {join(GENERATE_FILES.HOME, problem_basename)}_solver"
    )
    create_sol_command = (
        f"{join(GENERATE_FILES.HOME, problem_basename)}_solver -o "
        f"{join(problem_dir, GENERATE_FILES.DOMAIN_FILE)} -f {join(problem_dir, problem_basename)}.pddl "
    )
    if solver == GENERATE_FILES.LPG:
        create_sol_command += f"-n {sol_number} -out {join(output_dir, problem_basename)}.SOL -cputime {cpu_time} -v off"
    elif solver == GENERATE_FILES.FF:
        create_sol_command += f"-S {join(output_dir, problem_basename)}.SOL"
    delete_solver_command = f"rm {join(GENERATE_FILES.HOME, problem_basename)}_solver"

    if problem_basename != GENERATE_FILES.DOMAIN_FILE:
        os.system(copy_solver_command)
        os.system(create_sol_command)
        os.system(delete_solver_command)


def execute_create_xml(
    problem_path: str,
    solver_path: str,
    output_dir: str,
    solutions_dir: str = "./",
    sol_number: int = 4,
    solver: str = "LPG",
):

    problem_basename = basename(problem_path)
    problem_basename = problem_basename.split(".")[0]
    problem_dir = dirname(problem_path)
    copy_solver_command = (
        f"cp {solver_path} {join(GENERATE_FILES.HOME, problem_basename)}_solver"
    )
    create_xml_command = (
        f"{join(GENERATE_FILES.HOME, problem_basename)}_solver -o "
        f"{join(problem_dir, GENERATE_FILES.DOMAIN_FILE)} -f {join(problem_dir, problem_basename)}.pddl "
        "-input_plan {0} -n 1 -xml_addition_info -out {1} -v off"
    )
    delete_copy_command = "rm {0}"
    delete_solver_command = f"rm {join(GENERATE_FILES.HOME, problem_basename)}_solver"

    os.system(copy_solver_command)
    if problem_basename != "domain.pddl":
        if solver == "LPG":
            l = []
            for i in range(sol_number):
                l.append(i + 1)
            for el in l:

                f = join(solutions_dir, f"{problem_basename}.SOL_{el}.SOL")
                output_filename = f"xml-LPG-{problem_basename}{el}.SOL"
                o = join(output_dir, output_filename)

                if isfile(f) and not isfile(o):
                    clean_solution_file(f)
                    os.system(create_xml_command.format(f, o))
                    os.system(delete_copy_command.format(o))

        elif solver == "FF":
            f = join(solutions_dir, f"{problem_basename}.SOL")
            output_filename = f"xml-LPG-{problem_basename}.SOL"
            o = join(output_dir, output_filename)
            if isfile(f) and not isfile(o):
                os.system(create_xml_command.format(f, o))
                os.system(delete_copy_command.format(o))
        os.system(delete_solver_command)


def get_files(
    pddl_files_folder: str,
) -> list:
    e = ".pddl"
    files = [
        join(pddl_files_folder, f)
        for f in os.listdir(pddl_files_folder)
        if isfile(join(pddl_files_folder, f))
        and f != "domain.pddl"
        and (f.endswith(e) or f.endswith(e.upper()))
    ]
    return files


def get_solver_name(solver_path: str) -> str:
    if GENERATE_FILES.LPG in os.path.basename(solver_path).upper():
        return GENERATE_FILES.LPG
    elif GENERATE_FILES.FF in os.path.basename(solver_path).upper():
        return GENERATE_FILES.FF
    else:
        return ""


@click.group()
def run():
    pass


@run.command("sol")
@click.option(
    "--read-dir",
    "read_dir",
    type=click.STRING,
    help=HELPS.PDDL_FILE_FOLDER_SRC,
    prompt=True,
    required=True,
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
    "--target-dir",
    "target_dir",
    type=click.STRING,
    help=f"{HELPS.SOLUTIONS_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
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
def create_sol(read_dir, solver_path, target_dir, solutions, cpu_time, processors):
    files = get_files(read_dir)
    os.makedirs(target_dir, exist_ok=True)
    solver = get_solver_name(solver_path)
    if solver == "":
        return
    if processors == 1:
        for f in files:
            execute_create_solution(
                f, solver_path, target_dir, solutions, cpu_time, solver
            )
    else:
        with Pool(processors) as p:
            fun = partial(
                execute_create_solution,
                solver_path=solver_path,
                output_dir=target_dir,
                sol_number=solutions,
                cpu_time=cpu_time,
                solver=solver,
            )
            p.map(fun, files)


@run.command("xml")
@click.option(
    "--read-dir",
    "read_dir",
    type=click.STRING,
    help=HELPS.PDDL_FILE_FOLDER_SRC,
    prompt=True,
    required=True,
)
@click.option(
    "--solver-path",
    "solver_path",
    type=click.STRING,
    help=HELPS.LPG_SOLVER_FILE_SRC,
    prompt=True,
    required=True,
)
@click.option(
    "--solutions-dir",
    "solutions_dir",
    type=click.STRING,
    help=HELPS.SOLUTIONS_FOLDER_SRC,
    prompt=True,
    required=True,
)
@click.option(
    "--target-dir",
    "target_dir",
    type=click.STRING,
    help=f"{HELPS.XML_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
    required=True,
    prompt=True,
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
    "--processors",
    "processors",
    type=click.INT,
    help=HELPS.PROCESSORS_NUMBER,
    default=1,
    show_default=True,
)
@click.option(
    "--solutions-solver",
    "solver",
    type=click.Choice(["FF", "LPG"], case_sensitive=False),
    help=HELPS.SOLVER_SOLUTIONS,
    default=GENERATE_FILES.LPG,
    show_default=True,
)
def create_xmls(
    read_dir, solver_path, target_dir, solutions, processors, solutions_dir, solver
):
    files = get_files(read_dir)
    os.makedirs(target_dir, exist_ok=True)
    solver = get_solver_name(solver)
    if solver == "":
        return
    if processors == 1:
        for f in files:
            execute_create_xml(
                f, solver_path, target_dir, solutions_dir, solutions, solver
            )
    else:
        with Pool(processors) as p:
            fun = partial(
                execute_create_xml,
                solver_path=solver_path,
                output_dir=target_dir,
                solutions_dir=solutions_dir,
                sol_number=solutions,
                solver=solver,
            )
            p.map(fun, files)


if __name__ == "__main__":
    run()
