
import os
import logging
import sys
import yaml
import time
import sklearn

from src.config_builders.compile_params import create_pred_params
from src.utils.data_handler import  DataHandler
from src.utils.model import load_model, write_preds


def predict_pipeline(config_path, ts) -> None:
    logging.info("Loading params for validation ")
    
    with open(config_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    pred_params = create_pred_params(params)
    logging.info(f"Pipeline loaded the following parameters {pred_params}")

    test = DataHandler(pred_params.pred_params.load_split_path)
    features = set(test.data.columns) - set([pred_params.pred_params.target])
    
    x_test = test.data[features]
    y_test = test.data[pred_params.pred_params.target]
    
    model = load_model(pred_params.pred_params.load_model_path)
    logging.info(f"Successfully loaded model {pred_params.pred_params.load_model_path}")
    
    preds = model.predict(x_test)
    write_preds(preds, pred_params.pred_params.save_preds_path, f"predictions_{ts}.pred")

    logging.info(f"Model gave predictions, saved to predictions_{ts}.pred")

def validate(val_config_path: str, ts: float):
    ts = time.time()
    os.makedirs("data/outputs/logs", exist_ok=True)
    logfile = os.path.join("data/outputs/logs", str(ts) + ".log")
    print(f"Started logging into {logfile}")
    predict_pipeline(val_config_path, ts)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        validate(sys.argv[1], 0)
    else:
        validate(*sys.argv)