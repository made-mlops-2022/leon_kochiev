from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from app.schemas import Input

import pandas as pd
import requests
import pickle
import argparse

parser = argparse.ArgumentParser(
    prog="sender", description="send post request to predict listener"
)
parser.add_argument(
    "--path_to_model",
    default="https://storage.yandexcloud.net/leon/SVC_rbf_1.0_1667758675.4293056.pkl?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=YCAJEGLkF5mqZsXn_P9IDLez0%2F20221123%2Fru-central1%2Fs3%2Faws4_request&X-Amz-Date=20221123T035555Z&X-Amz-Expires=2592000&X-Amz-Signature=2664739F27E98F065AB125E37845B263CDDEF184A69DC6CC29E8442AF903EABE&X-Amz-SignedHeaders=host",
)
args = parser.parse_args()

app = FastAPI()
model = None


@app.exception_handler(RequestValidationError)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.on_event("startup")
async def download_model():
    global args
    global model
    url = args.path_to_model
    r = requests.get(url, allow_redirects=True)
    model = pickle.loads(r.content)


@app.post("/predict")
async def predict(request: Input):
    X = pd.DataFrame(
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    )
    for sample in request.data:
        X.loc[len(X.index)] = [
            sample.age,
            sample.sex,
            sample.cp,
            sample.trestbps,
            sample.chol,
            sample.fbs,
            sample.restecg,
            sample.thalach,
            sample.exang,
            sample.oldpeak,
            sample.slope,
            sample.ca,
            sample.thal,
        ]

    return {"target": model.predict(X).tolist()}


@app.get("/health")
async def status_model(status_code=status.HTTP_200_OK):
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="For some reason model is unavaliable :(",
        )
    else:
        return {"message": "Service is ready to accept requests!"}
