SMOTE_like_preprocessors = [
    'ADASYN', 'BorderlineSMOTE', #'ClusterCentroids', #'KMeansSMOTE', 
    #'NearMiss', 'RandomUnderSampler', 'RepeatedEditedNearestNeighbours', 
    'SMOTE', 'SMOTEENN', 'SMOTETomek', 'SVMSMOTE', #'TomekLinks', #'none', 'weighting'
]


feature_preprocessor = [
    'densifier', 'extra_trees_preproc_for_classification', 'fast_ica', # 'feature_agglomeration', 
    'kernel_pca', 'kitchen_sinks', 'liblinear_svc_preprocessor', 'no_preprocessing', 'nystroem_sampler', 'pca', 
    'polynomial', 'random_trees_embedding', 'select_percentile_classification', 'select_rates_classification', 
    'truncatedSVD'
]


EXPERIMENTS = [
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["SVMSMOTE"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "SVMSMOTE",
#         "METRIC": ["roc_auc"],
#         "SAMPLING_STRATEGY": True,
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["SVMSMOTE"],
#             "data_preprocessor": ["feature_type"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "SVMSMOTE",
#         "METRIC": ["roc_auc"],
#         "SAMPLING_STRATEGY": False,
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["weighting"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "weighting",
#         "METRIC": ["roc_auc"],
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["none"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "none",
#         "METRIC": ["roc_auc"],
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["SVMSMOTE"],
#             "data_preprocessor": ["feature_type"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "SVMSMOTE",
#         "METRIC": ["harmonic_mean_recall_2"],
#         "SAMPLING_STRATEGY": True,
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["weighting"],
#             "data_preprocessor": ["feature_type"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "weighting",
#         "METRIC": ["harmonic_mean_recall_2"],
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["none"],
#             "data_preprocessor": ["feature_type"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "none",
#         "METRIC": ["harmonic_mean_recall_2"],
#     },
#     {
#         "CLASSIFIER": "RF",
#     },
#     {
#         "CLASSIFIER": "SVMSMOTE+RF",
#     },
#     {
#         "CLASSIFIER": "weighting+RF",
#     },
]


EXPERIMENTS.extend([
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["SVMSMOTE"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "SVMSMOTE",
#         "METRIC": ["roc_auc"],
#         "SAMPLING_STRATEGY": True,
#         "CALLBACK": True,
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["weighting"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "weighting",
#         "METRIC": ["roc_auc"],
#         "CALLBACK": True,
#     },
#     {
#         "CLASSIFIER": "auto-sklearn",
#         "META": 25,
#         "INCLUDE": {
#             "balancing": ["none"],
#             "feature_preprocessor": feature_preprocessor
#         },
#         "DEFAULT": "none",
#         "METRIC": ["roc_auc"],
#         "CALLBACK": True,
#     },
])


EXPERIMENTS.extend([
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting"] + SMOTE_like_preprocessors,
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "SVMSMOTE",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "weighting",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "BorderlineSMOTE"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "BorderlineSMOTE",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "SMOTETomek"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "weighting",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "SMOTEENN"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "weighting",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "ADASYN"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "weighting",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "SMOTE"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "weighting",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
    {
        "CLASSIFIER": "auto-sklearn",
        "META": 25,
        "INCLUDE": {
            "balancing": ["none", "weighting", "SVMSMOTE"],
            "feature_preprocessor": feature_preprocessor
        },
        "DEFAULT": "SVMSMOTE",
        "METRIC": ["roc_auc"],
        "CALLBACK": False,
        "SAMPLING_STRATEGY": True
    },
])


def get_experiment_name(dct):
    classifier = dct["CLASSIFIER"]
    if classifier != "auto-sklearn":
        return classifier
    include = set(dct["INCLUDE"]["balancing"])
    meta = dct["META"]
    default = dct["DEFAULT"]
    metric = dct["METRIC"]
    if any(method in SMOTE_like_preprocessors for method in include):
        sampling_strategy = dct["SAMPLING_STRATEGY"]

    if set(SMOTE_like_preprocessors) <= include:
        include -= set(SMOTE_like_preprocessors)
        include.add("all_SMOTE_like")
    assert len(include) <= 3
    experiment_name = "+".join(sorted(include, key=lambda i: (i == "none", i == "weighting")))
    experiment_name = f"{experiment_name}-default={default}-META={meta}"
    if not (len(metric) == 1 and metric[0] == "roc_auc"):
        experiment_name = f"{experiment_name}-metric=[{','.join(metric)}]"
    if any(method in SMOTE_like_preprocessors for method in include) and not sampling_strategy:
        experiment_name = f"{experiment_name}-sampling_strategy={sampling_strategy}"
    if "CALLBACK" in dct and dct["CALLBACK"]:
        experiment_name = f"{experiment_name}-callback"
    return experiment_name


def get_experiments():
    return {
        get_experiment_name(experiment): experiment
        for experiment in EXPERIMENTS
    }


if __name__ == "__main__":
    for experiment in EXPERIMENTS:
        print(get_experiment_name(experiment))
