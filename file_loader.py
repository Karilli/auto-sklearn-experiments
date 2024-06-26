from enum import Enum
import pickle
import os
import json
from collections import defaultdict


if os.environ.get("USER") == "xkaras2":
    PREFIX = "/data/xkaras2/auto-sklearn"
elif os.environ.get("USER") == "karilli":
    PREFIX = "."
else:
    assert False


class Folders(str, Enum):
    TEMP = f"{PREFIX}/temp"
    VALIDATION_DIR = f"{PREFIX}/results/validation"
    MODELS_DIR = f"{PREFIX}/results/models"
    RUN_INFO_DIR = f"{PREFIX}/results/run_info"


def load_pkl(file):
    try:
        with open(file, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Cannot pickle load from file: {file}, error: {str(e)}")
        exit(1)


def load_json(file):
    try:
        with open(file, "rb") as f:
            return json.load(f)
    except:
        return {}


def dump_json(obj, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    try:
        with open(file, "w") as f:
            json.dump(obj, f)
    except:
        try:
            with open(file, "w") as f:
                json.dump(obj, f)
        except:
            pass
        exit(1)


def dump_pkl(obj, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    try:
        with open(file, "wb") as f:
            pickle.dump(obj, f)
    except:
        try:
            with open(file, "wb") as f:
                pickle.dump(obj, f)
        except:
            pass
        exit(1)


def get_results(DATASETS, EXPERIMENTS):
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    for dataset_name, _, _, ratio in DATASETS:
        for experiment_name in EXPERIMENTS:
            file = f"{Folders.VALIDATION_DIR}/{experiment_name}/{dataset_name}/{ratio:.2f}.json"
            res[dataset_name][ratio][experiment_name] = load_json(file)
    return res


def get_models(DATASETS, EXPERIMENTS):
    for dataset_name, _, _, ratio in DATASETS:
        for experiment_name in EXPERIMENTS:
            file = f"{Folders.MODELS_DIR}/{experiment_name}/{dataset_name}/{ratio:.2f}.pkl"
            yield load_pkl(file), dataset_name, ratio, experiment_name