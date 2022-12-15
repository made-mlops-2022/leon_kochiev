#/bin/bash

python3 sender.py http://localhost:8080/predict -d '{"data": [{"age":24, "sex":1, "cp":1, "trestbps":100, "chol":228, "fbs":0, "restecg":0, "thalach":100, "exang":0, "oldpeak":228.322, "slope":0, "ca":2, "thal":2}]}'