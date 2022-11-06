import os
import logging
import sys
import yaml
import time
# import hydra

from src.utils.data_handler import DataHandler
from src.utils.model import Solution
from src.config_builders.compile_params import create_train_params


def train_pipeline(config_path, ts):
    logging.info("Loading params for training ")
    with open(config_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train_params = create_train_params(params)
    logging.info(f"Pipeline loaded the following parameters {train_params}")

    data = DataHandler(train_params.paths.data_path)
    logging.info(f"Data loaded ")

    X_train, X_test, y_train, y_test = data.split(
        target_col=train_params.data_params.target,
        stratify=train_params.sampling_params.stratify,
        test_size=train_params.sampling_params.test_size,
        random_state_sampl=train_params.sampling_params.random_state_sampl,
    )
    logging.info(f"Data splitted ")

    model = Solution(
        exclude=[train_params.data_params.exclude],
        model=train_params.model_params.model,
        preprocessing=train_params.prepr_params.preprocessing,
        kernel=train_params.model_params.kernel,
        C=train_params.model_params.C,
        penalty=train_params.model_params.penalty,
        n_estimators=train_params.model_params.n_estimators,
        max_depth=train_params.model_params.max_depth,
    )
    logging.info(f"Model initialized")

    model.fit(X_train, y_train)
    logging.info(f"Model trained")

    preds = model.predict(X_test)
    preds_save_path = os.path.join(
        train_params.paths.save_preds_path, f"predictions_{ts}.pred"
    )
    with open(preds_save_path, "w") as file:
        for item in preds:
            file.write("%s\n" % item)

    logging.info(f"Model gave predictions, saved to {preds_save_path}")

    if train_params.model_params.model == "LogisticRegression":
        filename = f"{train_params.model_params.model}_{train_params.model_params.penalty}_{train_params.model_params.C}_{ts}.pkl"
    elif train_params.model_params.model == "RandomForestClassifier":
        filename = f"{train_params.model_params.model}_{train_params.model_params.n_estimators}_{train_params.model_params.max_depth}_{ts}.pkl"
    elif train_params.model_params.model == "SVC":
        filename = f"{train_params.model_params.model}_{train_params.model_params.kernel}_{train_params.model_params.C}_{ts}.pkl"
        
    model.save(train_params.paths.save_model_path, filename)
    model_path = os.path.join(train_params.paths.save_model_path, filename)
    logging.info(f"Model saved to {model_path}")
    
    return preds, y_test

# @hydra.main(version_base=None, config_path="/home/nullkatar/datasets/MADE/mlops/hw1/configs")
def train(train_config_path: str):
    ts = time.time()
    logfile = os.path.join("data/outputs/logs", str(ts) + ".log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    print(f"Started logging into {logfile}")
    x_test, y_test = train_pipeline(train_config_path, ts)

    return ts, x_test, y_test


if __name__ == "__main__":
    train(sys.argv[1])
