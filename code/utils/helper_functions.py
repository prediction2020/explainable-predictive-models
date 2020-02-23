"""
File name: helper_functions.py
Author: Esra Zihni
Date created: 15.02.2018

This file contains helper functions for other scripts.
"""

import catboost as cat
import innvestigate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from keras import activations
from scipy.stats import iqr
from sklearn import preprocessing
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from vis.utils import utils


def subsample(X, y, subsampling_type: str):
    """
    If subsampling type is defined as 'random', randomly sub-samples the given 
    data to yield equal number of label classes. Otherwise if subsampling type 
    is defined as 'none', returns original data. 

    :param X: Data input variables
    :param y: Data labels
    :param subsampling_type: Subsampling method to be used.
    :return: Subsampled data input variables and labels.
    """
    df = pd.concat([X, y], axis=1)
    label = list(df)[-1]

    if subsampling_type == "random":
        df_bad = df[(df[label] == 1)]
        df_good = df[(df[label] == 0)]

        df_sub = pd.concat([df_good.sample(len(df_bad.index), random_state=21), df_bad])
        X_new, y_new = df_sub.drop(label, axis=1), df_sub[[label]]

    elif subsampling_type == "none":
        X_new, y_new = X, y

    return X_new, y_new


def predict_probability(data, model, model_name: str):
    """
    Returns prediction probabilities estimated for the given dataset and model.

    :param data: Training or test data.
    :param model: Trained model.
    :param model_name: Name of the trained model.
    :return: Prediction probabilities
    """
    if model_name == "MLP":
        output_activation = model.get_config()["layers"][-1]["config"]["activation"]
        if output_activation == "softmax":
            probs = model.predict_proba(data)
        else:
            probs = model.predict(data)
    else:
        probs = model.predict_proba(data).T[1]

    return probs


def predict_class(data, model, model_name: str):
    """
    Returns predicted classes for the given dataset and model.

    :param data: Training or test data.
    :param model: Trained model.
    :param model_name: Name of the trained model.
    :return: Predicted classes
    """
    if model_name == "MLP":
        output_activation = model.get_config()["layers"][-1]["config"]["activation"]
        if output_activation == "softmax":
            probs = model.predict(data)
            preds = probs.argmax(axis=-1)
        else:
            preds = model.predict_classes(data)
    else:
        preds = model.predict(data).astype("float64")

    return preds


def calc_perf_score(data, labels, model, model_name: str, score_name: str):
    """
    Returns performance based on the given performance measure for the
    given data and model.

    :param data: Training or test data input variables.
    :param labels: Training or test labels.
    :param model: Trained model.
    :param model_name: Name of the trained model.
    :param score_name: Name of the performance measure.
    :return: Performance score
    """
    if isinstance(labels, pd.DataFrame):
        labels.iloc[:, 0] = labels.iloc[:, 0].astype("float64")

    if score_name == "AUC":
        probs = predict_probability(data, model, model_name)
        score = roc_auc_score(labels, probs)

    else:
        preds = predict_class(data, model, model_name)
        if score_name == "accuracy":
            score = accuracy_score(labels, preds)
        elif score_name == "f1":
            score = f1_score(labels, preds, pos_label=1)
        elif score_name == "average_class_accuracy":
            recall = recall_score(labels, preds, average=None)
            score = 2 / (1 / recall[0] + 1 / recall[1])

    return score


def calc_shap_values(dataset, model):
    """
    Calculates Shapley values for the given training set and tree boosting model.

    :param dataset: Training or test data.
    :param model: Trained tree boosting model.
    :return: Shapley values
    """
    explainer = shap.TreeExplainer(model.best_model)
    cat_features = [
        list(model.X_tr).index(dataset.cat_preds[i])
        for i in range(len(dataset.cat_preds))
    ]

    shap_values = explainer.shap_values(
        cat.Pool(model.X_tr, model.y_tr, cat_features=cat_features)
    )

    # Calculate average over samples (patients)
    shap_values_mean_over_samples = np.mean(shap_values, axis=0)

    return shap_values_mean_over_samples


def calc_deep_taylor_values(model):
    """
    Calculates deep taylor decomposition values for the given training set and 
    Multilayer Perceptron (MLP) model.

    :param dataset: Training or test data.
    :param model: Trained MLP model.
    :return: Deep taylor values
    """
    # Predict training and test probabilities
    test_probs = predict_probability(model.X_te, model.best_model, "MLP")
    train_probs = predict_probability(model.X_tr, model.best_model, "MLP")

    # Set last layer activation to linear. If this swapping is not done, the
    # results might be suboptimal
    model.best_model.layers[-1].activation = activations.linear
    stripped_model = utils.apply_modifications(model.best_model)

    # Calculate class weights
    train_input_weights = train_probs
    train_input_weights[np.where(model.y_tr == 0)] = (
        1 - train_input_weights[np.where(model.y_tr == 0)]
    )

    # Get last layer index
    class_idx = 0  # if the activation of last layer was sigmoid
    last_layer_idx = utils.find_layer_idx(model.best_model, "dense_2")

    # Get the input the model was trained on
    seed_input = model.X_tr.values
    # The deep taylor is bounded to a range which should be defined based on 
    # the input range:
    input_range = [min(seed_input.flatten()), max(seed_input.flatten())]

    # Calculate global gradients of all patients (deep taylor)
    gradient_analyzer = innvestigate.create_analyzer(
        "deep_taylor.bounded",  # analysis method identifier
        stripped_model,  # model without softmax output
        low=input_range[0],
        high=input_range[1],
    )

    analysis = gradient_analyzer.analyze(seed_input)

    # Calculate score based average
    t_analysis = np.transpose(analysis, (1, 0))
    train_input_weights_s = np.squeeze(train_input_weights)
    score_avg_analysis = np.expand_dims(
        np.dot(t_analysis, train_input_weights_s), axis=0
    )

    return score_avg_analysis


def plot_performance(scores, model_names, sub_type, path):
    """
    Plots performance scores as a matplotlib errorbar plot. Saves the created
    plot as a .png file.

    :param scores: Training and test performance scores.
    :param model_names: Names of trained models.
    :param sub_type: Type of subsampling used on the training data.
    :param path: Path to save plot.
    """
    perfs = list(scores.keys())
    model_count = len(model_names)

    fig, ax = plt.subplots(
        1,
        len(perfs),
        sharey=False,
        sharex=True,
        figsize=(len(perfs) * 7, 5),
        squeeze=False,
    )
    plt.style.use("seaborn-notebook")
    for i, perf in enumerate(perfs):
        score = scores[perf]
        median_tr = [np.median(score[m], axis=0)[0] for m in model_names]
        iqr_tr = [iqr(score[m], axis=0)[0] for m in model_names]
        median_te = [np.median(score[m], axis=0)[1] for m in model_names]
        iqr_te = [iqr(score[m], axis=0)[1] for m in model_names]

        ax[0, i].errorbar(
            range(model_count),
            median_tr,
            yerr=iqr_tr,
            marker="o",
            markersize="12",
            ls="None",
            alpha=0.7,
            label="training "
            + perf.replace("average", "avg")
            .replace("_", " ")
            .replace("accuracy", "acc"),
        )
        ax[0, i].errorbar(
            range(model_count),
            median_te,
            yerr=iqr_te,
            marker="o",
            markersize="12",
            ls="None",
            alpha=0.7,
            label="test "
            + perf.replace("average", "avg")
            .replace("_", " ")
            .replace("accuracy", "acc"),
        )
        ax[0, i].legend(loc="lower left")
        ax[0, i].set_title(perf.replace("_", " "), size=11)
        ax[0, i].tick_params(axis="both", labelsize=10)
        if perf == "AUC":
            ax[0, i].set_ylim(0.5, 1)
        else:
            ax[0, i].set_ylim(0.3, 1)

    model_names_ticks = [
        m.replace("Catboost", "Tree Boosting").replace("ElasticNet", "Elastic Net")
        for m in model_names
    ]

    ax[0, 0].set_ylabel("Value", size=16)
    plt.xticks(range(model_count), model_names_ticks)
    plt.suptitle("Final Performance Scores", size=12, y=1.04)
    plt.tight_layout()
    fig.savefig(
        f"{path}/performance_scores_{sub_type}_subsampling.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_features_rating(values, sub_type, path):
    """
    Plots feature importance values as a seaborn barplot. Saves the created
    plot as a .png file.

    :param values: Feature importance values.
    :param sub_type: Type of subsampling used on the training data.
    :param path: Path to save plot.
    """
    models = list(values.keys())
    sns.set(style="white", rc={"xtick.labelsize": 8, "ytick.labelsize": 10})

    fig, ax = plt.subplots(
        1,
        len(models),
        sharey=True,
        sharex=True,
        figsize=(1.5 * len(models), 3),
        squeeze=False,
    )
    for i, mdl in enumerate(models):
        value = values[mdl].copy()
        value[:] = preprocessing.normalize(value)
        sns.barplot(
            data=abs(value),
            orient="h",
            ci="sd",
            errwidth=0.9,
            estimator=np.mean,
            ax=ax[0, i],
        )

        if mdl == "Catboost":
            ax[0, i].set_xlabel("|SHAP values|", size=9)
        elif mdl == "MLP":
            ax[0, i].set_xlabel("|deep Taylor values|", size=9)
        else:
            ax[0, i].set_xlabel("|weights|", size=9)
        ax[0, i].set_title(
            mdl.replace("Catboost", "Tree Boosting").replace(
                "Elasticnet", "Elastic Net"
            ),
            size=10,
        )
    plt.suptitle("Clinical Features Importance Rating", size=12, y=1.02)

    fig.savefig(
        f"{path}/clinical_features_rating_{sub_type}_subsampling.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
