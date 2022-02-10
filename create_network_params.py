import os
from params_generator import ParamsGenerator
from os.path import join
import json
from constants import HELPS, PARAMS_GEN, FILENAMES
import click


@click.command()
@click.option(
    "--target-dir",
    "target_dir",
    type=click.STRING,
    required=True,
    prompt=True,
    help=f"{HELPS.PARAMS_TEMPLATE_DIR_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
@click.option(
    "--num-experiments",
    "num_experiments",
    type=click.INT,
    help=HELPS.PARAMETERS_NUMBER,
    default=1,
    show_default=True,
)
def run(target_dir, num_experiments):
    os.makedirs(target_dir, exist_ok=True)

    model_name = PARAMS_GEN.DEFAULT_MODEL_NAME

    p = ParamsGenerator(model_name)

    params_list = p.generate(num_experiments)
    for i in range(len(params_list)):
        params_file = join(target_dir, f"{FILENAMES.PARAMS_TEMPLATE_FILENAME}_{i}.json")
        with open(params_file, "w") as f:
            json.dump(dict(params_list[i]), f, indent=4)
            f.close()


if __name__ == "__main__":
    run()
