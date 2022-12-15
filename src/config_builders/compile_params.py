from dataclasses import dataclass

from .model_params import ModelParams, PreprParams
from .paths import Paths
from .data_params import DataParams
from .sampling_params import SamplingParams
from .validation_params import ValidationParams
from .prediction_params import PredictionParams

# https://habr.com/ru/post/539600/
from marshmallow_dataclass import class_schema

@dataclass()
class TrainParams:
    paths: Paths
    sampling_params: SamplingParams
    prepr_params: PreprParams
    model_params: ModelParams
    data_params: DataParams

@dataclass()
class ValParams:
    val_params: ValidationParams
    
@dataclass()
class PredParams:
    pred_params: PredictionParams
    
TrainParamsSchema = class_schema(TrainParams)
ValParamsSchema = class_schema(ValParams)
PredParamsSchema = class_schema(PredParams)

def create_train_params(config) -> TrainParams:
    schema = TrainParamsSchema()
    return schema.load(config)

def create_val_params(config) -> ValParams:
    schema = ValParamsSchema()
    return schema.load(config)

def create_pred_params(config) -> PredParams:
    schema = PredParamsSchema()
    return schema.load(config)