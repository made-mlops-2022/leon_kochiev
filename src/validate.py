
import os
import logging
import sys
import yaml
import time
import sklearn

from src.config_builders.compile_params import create_val_params
from src.utils.data_handler import  DataHandler, load_preds
from src.utils.validator import Validaror


def validate_pipeline(config_path, ts, preds, y_test) -> None:
    logging.info("Loading params for validation ")
    with open(config_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    val_params = create_val_params(params)
    logging.info(f"Pipeline loaded the following parameters {val_params}")

    if preds is None:
        y_test = DataHandler(val_params.val_params.load_split_path).data[val_params.val_params.target]

        #taking latest prediction
        ts = val_params.val_params.load_preds_path.split('_')[-1][:-5]
        logging.info(f"Validating latest prediction at timestamp {ts}")
        preds = load_preds(val_params.val_params.load_preds_path)

    validator = Validaror()
    out = validator.validate(
        [
            getattr(sklearn.metrics, i)
            for i in [
                val_params.val_params.metrics,
            ]
        ],
        preds,
        y_test,
    )
    logging.info(f"The final metrics are {out}")

def validate(val_config_path: str, ts: float, X_test, y_test):
    ts = time.time()
    logfile = os.path.join("data/outputs/logs", str(ts) + ".log")
    print(f"Started logging into {logfile}")
    validate_pipeline(val_config_path, ts, X_test, y_test)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        validate(sys.argv[1], 0, None, None)
    else:
        validate(*sys.argv)