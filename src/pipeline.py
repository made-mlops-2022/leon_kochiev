import sys

from train import train
from validate import validate

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ts, X_test, y_test = train(sys.argv[1])
        validate(sys.argv[2], ts, X_test, y_test)
