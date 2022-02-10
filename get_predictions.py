from keras.models import load_model
from attention_extraction_layers import AttentionWeights, ContextVector
from plan_generator import PlanGenerator
from utils_functions import load_from_pickles, save_file
from create_neural_network import get_model_predictions
import os
import click
from constants import HELPS, ERRORS, FILENAMES


@click.command()
@click.option(
    "--model",
    "model_path",
    required=True,
    prompt=True,
    type=click.STRING,
    help=HELPS.MODEL_SRC,
)
@click.option(
    "--read-test-plans-dir",
    "read_test_plans_dir",
    required=True,
    prompt=True,
    type=click.STRING,
    help=HELPS.TEST_PLANS_DIR_SRC,
)
@click.option(
    "--read-dict-dir",
    "read_dict_dir",
    required=True,
    prompt=True,
    type=click.STRING,
    help=HELPS.DICT_FOLDER_SRC,
)
@click.option(
    "--max-plan-perc",
    "max_plan_perc",
    default=0.7,
    type=click.FLOAT,
    help=HELPS.MAX_PLAN_PERCENTAGE,
    show_default=True,
)
@click.option(
    "--plan-perc",
    "plan_perc",
    required=True,
    type=click.FloatRange(0, 1),
    help=HELPS.PLAN_PERCENTAGE,
    prompt=True,
)
@click.option(
    "--max-plan-dim",
    "max_plan_dim",
    required=True,
    type=click.INT,
    prompt=True,
    help=HELPS.MAX_PLAN_LENGTH,
)
@click.option(
    "--batch-size",
    "batch_size",
    type=click.INT,
    default=64,
    show_default=True,
    help=HELPS.BATCH_SIZE,
)
@click.option(
    "--target-dir",
    "target_dir",
    type=click.STRING,
    required=True,
    prompt=True,
    help=f"{HELPS.PRED_DIR_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
def run(
    read_dict_dir,
    read_test_plans_dir,
    model_path,
    max_plan_perc,
    plan_perc,
    max_plan_dim,
    batch_size,
    target_dir,
):
    try:
        model = load_model(
            model_path,
            custom_objects={
                "AttentionWeights": AttentionWeights,
                "ContextVector": ContextVector,
            },
        )
        print(
            ERRORS.MSG_LOAD_MODEL_OK.format(
                os.path.basename(model_path), os.path.dirname(model_path)
            )
        )
        print(model.summary())
    except OSError as e:
        print(e)
        print(
            ERRORS.MSG_ERROR_LOAD_MODEL.format(
                os.path.basename(model_path), os.path.dirname(model_path)
            )
        )
        model = None

    filenames = os.listdir(read_test_plans_dir)
    [dizionario, dizionario_goal] = load_from_pickles(
        read_dict_dir, [FILENAMES.ACTION_DICT_FILENAME, FILENAMES.GOALS_DICT_FILENEME]
    )
    test_plans = load_from_pickles(read_test_plans_dir, filenames)

    max_dim = int(max_plan_perc * max_plan_dim)

    if (
        model is None
        or test_plans is None
        or dizionario is None
        or dizionario_goal is None
    ):
        print(ERRORS.MSG_ERROR_CREATE_PREDICTIONS)
    else:
        for i, plans in enumerate(test_plans):
            gen = PlanGenerator(
                plans,
                dizionario,
                dizionario_goal,
                batch_size,
                max_dim,
                plan_perc,
                shuffle=False,
            )
            filename = filenames[i]
            y_true, y_pred = get_model_predictions(model, gen)
            res_dict = dict()
            res_dict["y_test_true"] = y_true
            res_dict["y_test_pred"] = y_pred

            os.makedirs(target_dir, exist_ok=True)
            save_file(res_dict, target_dir, f"{filename}.pkl")


if __name__ == "__main__":
    run()
