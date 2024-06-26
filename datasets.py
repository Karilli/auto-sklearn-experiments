from sklearn.datasets import fetch_openml
from imblearn.datasets import make_imbalance
import numpy as np


IDS = [
    # ("breast-w", 15),
    ## ("credit-approval", 29),
    ## ("credit-g", 31),
    ("diabetes", 37),
    ## ("sick", 38),
    # ("spambase", 44),
    ## ("tic-tac-toe", 50),
    ## ("electricity", 151),
    ## ("vowel", 307),
    # ("pc4", 1049),
    ("pc3", 1050),
    ("JM1", 1053),
    ("KC2", 1063),
    ("kc1", 1067),
    ("pc1", 1068),
    ## ("bank-marketing", 1461),
    ("blood-transfusion-service-center", 1464),
    ## ("ilpd", 1480),
    ("madelon", 1485),
    ## ("nomao", 1486),
    ("ozone-level-8hr", 1487),
    # ("phoneme", 1489),
    ("qsar-biodeg", 1494),
    ## ("adult", 1590),
    ("Bioresponse", 4134),
    ## ("cylinder-bands", 6332),
    ## ("dresses-sales", 23381),
    ## ("numerai28.6", 23517),
    ## ("churn", 40701),
    # ("wilt", 40983),
    ("climate-model-simulation-crashes", 40994),
]


RATIOS = [
    "original",
    0.15, 
    0.10, 
    0.05, 
]


def get_datasets(SEED=0):
    res = []

    for dataset_name, dataset_id in IDS:
        new_name = f"{dataset_name}(id={dataset_id})"

        try:
            data = fetch_openml(data_id=dataset_id, parser="auto", as_frame=True)
        except:
            print(f"Couldn't fetch dataset {dataset_name} with id {dataset_id}.")
            continue

        numerical_features = data.frame.select_dtypes(include=['int64', 'float64']).columns
        target_names = data.target_names

        if not (set(numerical_features) == set(data.feature_names) and len(target_names) == 1):
            print(f"Dataset {dataset_name} with id {dataset_id} contains non-numerical features.")
            continue

        if 50_000 < data.frame.shape[0]:
            print(f"Dataset {dataset_name} with id {dataset_id} has more then 50 000 instances.")
            continue

        if data.frame.isna().any().any():
            amount = data.frame.isna().sum().sum() / data.frame.shape[0]
            if 0.05 < amount:
                print(f"Dataset {dataset_name} with id {dataset_id} contains NaN - {amount}.")
                continue
            data.frame.dropna(inplace=True)

        target = target_names[0]
        assert len(data.frame[target].unique()) == 2, dataset_name

        X = data.frame.drop(columns=[target])
        (l1, l2), (c1, c2) = np.unique(data.frame[target], return_counts=True)
        (c1, l1), (c2, l2) = sorted(((c1, l1), (c2, l2)))
        y = data.frame[target] == l1

        if "original" in RATIOS:
            res.append((new_name, X, y, c1/c2))

        for ratio in sorted((r for r in RATIOS if r != "original"), reverse=True):
            try:
                X, y = make_imbalance(X, y, sampling_strategy={0: c2, 1: int(c2 * ratio)}, random_state=SEED)
                res.append((new_name, X, y, ratio))
            except ValueError as e:
                if "With under-sampling methods, the number of samples in a class should be less or equal to the original number of samples." in str(e):
                    # print(f"Error with RandomUndersampler on dataset {dataset_name} {Counter(y)}, ratio={ratio}")
                    continue
                raise e
    return res
