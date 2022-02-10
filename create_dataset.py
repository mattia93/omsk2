import os
import numpy as np
import utils
import oneHot_deep
import click
from utils_functions import save_file, create_table, create_plot
from constants import CREATE_DATASET, HELPS, FILENAMES
from os.path import join


def create_dictionary(plans: list, oneHot: bool = True):
    dictionary = oneHot_deep.create_dictionary(plans)
    dictionary = oneHot_deep.shuffle_dictionary(dictionary)
    if oneHot:
        oneHot_deep.completa_dizionario(dictionary)
    return dictionary


def create_dictionary_goals_fixed(plans):
    goals = []
    for p in plans:
        if p.goals not in goals:
            goals.append(p.goals)
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


def create_dictionary_goals_not_fixed(plans):
    goals = []
    for p in plans:
        for fact in p.goals:
            if fact not in goals:
                goals.append(fact)
    dizionario_goal = oneHot_deep.create_dictionary_goals(goals)
    dizionario_goal = oneHot_deep.shuffle_dictionary(dizionario_goal)
    oneHot_deep.completa_dizionario(dizionario_goal)
    return dizionario_goal


def compute_bins_number(min_value: int, max_value: int, max_bin_number: int) -> int:
    if max_value - min_value + 1 < max_bin_number:
        return max_value - min_value + 1
    else:
        return max_bin_number


def create_quantile_table(
    values: list,
    max_nbins: int,
    table_title: str = "Table",
    headers: list = CREATE_DATASET.TABLE_HEADERS,
    stats_file: str = None,
) -> int:
    rows = list()
    row = list()
    for i in range(len(headers)):
        row.append(np.quantile(values, i / 4))
    rows.append(row)
    nbins = compute_bins_number(row[0], row[4], max_nbins)
    table = create_table(table_title, headers, rows)
    for row in table:
        print(row)
    if stats_file is not None:
        with open(stats_file, "a") as af:
            for row in table:
                af.write(f"{row}\n")
            af.write("\n")
    return int(nbins)


def print_plans_stat(
    plans: list, nbins: int = 10, save_graph: str = None, stats_file: str = None
) -> None:
    print(CREATE_DATASET.PLANS_NUMBER.format(len(plans)))
    if stats_file is not None:
        with open(stats_file, "a") as af:
            af.write(f"{CREATE_DATASET.PLANS_NUMBER.format(len(plans))}\n")
    plans_len = list()
    for p in plans:
        plans_len.append(len(p.actions))
    nbins = create_quantile_table(
        plans_len, nbins, CREATE_DATASET.PLANS_TABLE_TITLE, stats_file=stats_file
    )
    create_plot(plot_type="hist", target_dir=save_graph, input=plans_len, nbins=nbins)


def print_action_distrib(
    plans: list, save_graph: str = None, nbins: int = 10, stats_file: str = None
) -> None:
    freq_action_dict = dict()
    for p in plans:
        for a in p.actions:
            a = a.name
            if a in freq_action_dict.keys():
                freq_action_dict[a] += 1
            else:
                freq_action_dict[a] = 1

    print(CREATE_DATASET.ACTIONS_NUMBER.format(len(freq_action_dict)))
    if stats_file is not None:
        with open(stats_file, "a") as af:
            af.write(f"{CREATE_DATASET.ACTIONS_NUMBER.format(len(freq_action_dict))}\n")
    v = list(freq_action_dict.values())
    nbins = create_quantile_table(
        v, nbins, CREATE_DATASET.ACTIONS_TABLE_TITLE, stats_file=stats_file
    )
    create_plot(plot_type="hist", input=v, nbins=nbins, target_dir=save_graph)


def print_goal_distrib(
    plans: list, save_graph: str = None, nbins: int = 10, stats_file: str = None
):
    goals_dict = dict()
    for p in plans:
        for g in p.goals:
            if g in goals_dict.keys():
                goals_dict[g] = goals_dict[g] + 1
            else:
                goals_dict[g] = 1
    print(CREATE_DATASET.GOALS_NUMBER.format(len(goals_dict)))
    if stats_file is not None:
        with open(stats_file, "a") as af:
            af.write(f"{CREATE_DATASET.GOALS_NUMBER.format(len(goals_dict))}\n")
    v = list(goals_dict.values())
    nbins = create_quantile_table(
        v, nbins, CREATE_DATASET.GOALS_TABLE_TITLE, stats_file=stats_file
    )
    create_plot(plot_type="hist", input=v, nbins=nbins, target_dir=save_graph)


@click.command()
@click.option(
    "--read-dir",
    "read_dir",
    prompt=True,
    required=True,
    type=click.STRING,
    help=HELPS.XML_FOLDER_SRC,
)
@click.option(
    "--target-dir",
    "target_dir",
    prompt=True,
    required=True,
    type=click.STRING,
    help=f"{HELPS.PLANS_AND_DICT_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
@click.option("--onehot", is_flag=True, default=False, help=HELPS.ONEHOT_FLAG)
@click.option(
    "--plots-dir",
    "plots_dir",
    prompt=True,
    required=True,
    type=click.STRING,
    help=f"{HELPS.PLOTS_FOLDER_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
@click.option(
    "--save-stats",
    "save_stats",
    is_flag=True,
    help=HELPS.SAVE_STATS_FLAG,
)
def run(read_dir, target_dir, plots_dir, onehot, save_stats):

    os.makedirs(target_dir, exist_ok=True)

    plans = utils.get_all_plans(read_dir)
    dizionario = create_dictionary(plans, onehot)
    dizionario_goal = create_dictionary_goals_not_fixed(plans)

    save_file(plans, target_dir, FILENAMES.PLANS_FILENAME)
    save_file(dizionario, target_dir, FILENAMES.ACTION_DICT_FILENAME)
    save_file(dizionario_goal, target_dir, FILENAMES.GOALS_DICT_FILENEME)

    os.makedirs(plots_dir, exist_ok=True)
    stats_file = None
    if save_stats:
        stats_file = join(plots_dir, FILENAMES.STATS_FILENAME)
        open(stats_file, "w")
    print_plans_stat(
        plans,
        nbins=CREATE_DATASET.NBINS,
        save_graph=join(plots_dir, FILENAMES.PLOT_LENGTH_FILENAME),
        stats_file=stats_file,
    )
    print_action_distrib(
        plans,
        nbins=CREATE_DATASET.NBINS,
        save_graph=join(plots_dir, FILENAMES.PLOT_ACTIONS_FILENAME),
        stats_file=stats_file,
    )
    print_goal_distrib(
        plans,
        nbins=CREATE_DATASET.NBINS,
        save_graph=join(plots_dir, FILENAMES.PLOT_GOALS_FILENAME),
        stats_file=stats_file,
    )


if __name__ == "__main__":
    np.random.seed(47)
    run()
