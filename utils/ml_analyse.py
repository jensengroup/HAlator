import pandas as pd
import numpy as np
import joblib

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


from scipy.stats import spearmanr

from scipy.stats import pearsonr
import math

import matplotlib.pyplot as plt
import seaborn as sns


import operator
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy.stats import uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import random


import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

from pprint import pprint
import time

from itertools import islice

# ----------------- ML ANALYSIS -----------------


def prepare_data(df, descriptor_col, target_col):
    X_data = []
    y_data = []

    for index, row in df.iterrows():
        for i in range(len(row.atom_index)):
            # if row['lst_pka_sp_lfer'][i] == float('inf')
            if row[target_col][i] >= 70:
                continue
            else:
                X_data.append(row[descriptor_col][i])
                y_data.append(row[target_col][i])

    lst_len_atom_index = [len(a_idx) for a_idx in df.atom_index.values]
    df_prep_all = pd.DataFrame(X_data)
    df_prep_all["label"] = y_data
    df_regression = df_prep_all.sample(frac=0.9, random_state=42).reset_index(drop=True)
    df_unseen_regression = df_prep_all.drop(df_prep_all.index).reset_index(drop=True)
    return df_prep_all, df_regression, df_unseen_regression, lst_len_atom_index


def create_dict_from_testset(df_testset, name="", regression=True):
    dict_comp_name = {}
    for unique_name in df_testset.names.unique():
        atom_index_list = df_testset.loc[df_testset["names"] == unique_name][
            "atom_index"
        ].tolist()  # f"comp_name == {unique_names}")['atom_index'].tolist()
        label_list = df_testset.loc[df_testset["names"] == unique_name][
            "label"
        ].tolist()
        pred_list = df_testset.loc[df_testset["names"] == unique_name]["pred"].tolist()
        if regression:
            pred_list = [round(pred, 4) for pred in pred_list]
            dict_comp_name[unique_name] = {
                f"atom_indices_pred_{name}": atom_index_list,
                f"lst_label_{name}": label_list,
                f"lst_pka_pred_{name}": pred_list,
                f"pka_min_pred_{name}": min(pred_list),
                f"error_pred_vs_calc_{name}": [
                    round(abs(pred - label), 4)
                    for pred, label in zip(pred_list, label_list)
                ],
                "train_test": "test",
            }
        else:
            pred_list = [int(pred) for pred in pred_list]
            dict_comp_name[unique_name] = {
                f"atom_indices_pred_{name}": atom_index_list,
                f"lst_label_{name}": label_list,
                f"atom_lowest_pred_{name}": pred_list,
                "train_test": "test",
            }

    return dict_comp_name


def add_pred_to_dataset(dataset, dict_comp_name):
    dataset = dataset.copy()
    for key, value in dict_comp_name.items():
        idx = dataset[dataset["names"] == key].index
        for subkey, subvalue in value.items():
            if subkey not in dataset.columns:
                dataset[subkey] = None  # Initialize the column with None (NaN) values
            dataset.loc[idx, subkey] = dataset.loc[idx, subkey].apply(
                lambda x: subvalue
            )
    return dataset


def correct_pred_results(dataset, name, regression=True):
    dataset = dataset.copy()
    for index, row in dataset.query('train_test == "test"').iterrows():
        col1 = row["atom_indices"]
        col2 = row[f"atom_indices_pred_{name}"]

        if regression:
            col3 = row[f"lst_pka_pred_{name}"]
            col4 = row[f"error_pred_vs_calc_{name}"]
        else:
            col3 = row[f"atom_lowest_pred_{name}"]

        # Check if col2 is equal to col1
        if col2 == col1:
            temp_list = (
                col3  # If they are equal, use col3 as is as all sites are predicted
            )
            if regression:
                temp_list_error = col4
        else:
            # Create a temporary list with inf for missing items
            temp_list = [
                col3[col2.index(item)] if item in col2 else np.inf for item in col1
            ]
            if regression:
                temp_list_error = [
                    col4[col2.index(item)] if item in col2 else np.inf for item in col1
                ]

        if regression:
            if f"atom_lowest_pred_{name}" not in dataset.columns:
                dataset[f"atom_lowest_pred_{name}"] = None
                dataset[f"atom_lowest_pred_{name}1"] = None
                dataset[f"atom_lowest_pred_{name}2"] = None
            dataset.at[index, f"atom_lowest_pred_{name}"] = (
                [-1 for _ in temp_list]
                if row[f"pka_min_pred_{name}"] == float("inf")
                else [
                    (
                        1
                        if i == row[f"pka_min_pred_{name}"]
                        else -1 if i == float("inf") else 0
                    )
                    for i in temp_list
                ]
            )
            dataset.at[index, f"atom_lowest_pred_{name}1"] = (
                [-1 for _ in temp_list]
                if row[f"pka_min_pred_{name}"] == float("inf")
                else [
                    (
                        1
                        if abs(i - row[f"pka_min_pred_{name}"]) <= 1
                        or i == row[f"pka_min_pred_{name}"]
                        else -1 if i == float("inf") else 0
                    )
                    for i in temp_list
                ]
            )
            dataset.at[index, f"atom_lowest_pred_{name}2"] = (
                [-1 for _ in temp_list]
                if row[f"pka_min_pred_{name}"] == float("inf")
                else [
                    (
                        1
                        if abs(i - row[f"pka_min_pred_{name}"]) <= 2
                        or i == row[f"pka_min_pred_{name}"]
                        else -1 if i == float("inf") else 0
                    )
                    for i in temp_list
                ]
            )
            dataset.at[index, f"lst_pka_pred_{name}"] = temp_list
            dataset.at[index, f"error_pred_vs_calc_{name}"] = temp_list_error
        else:
            dataset.at[index, f"atom_lowest_pred_{name}"] = [
                i if i != float("inf") else -1 for i in temp_list
            ]

    return dataset


def create_df_for_ml(
    df_source,
    col_name="names",
    feature_col="descriptor_vector",
    label_col="lst_pka_lfer",
    atom_index_col="atom_indices",
    regression=True,
):
    """_summary_

    Args:
        test_indices (_type_): _description_
        df_source (_type_): _description_
        feature_col (str, optional): _description_. Defaults to 'descriptor_vector_conf20'.
        label_col (str, optional): _description_. Defaults to 'lst_pka_sp_lfer'.
        atom_index_col (str, optional): _description_. Defaults to 'atom_indices_conf20'.

    Returns:
        _type_: _description_
    """
    X_test = []
    y_test = []
    atom_idx_test = []
    idx_test = []
    comp_names = []

    df_loc = df_source[[col_name, feature_col, label_col, atom_index_col]]

    for idx, row in df_loc.iterrows():
        for desc, label, atom_idx in zip(
            row[feature_col], row[label_col], row[atom_index_col]
        ):
            if regression:
                if label == float("inf"):
                    continue
            elif not regression:
                if label == -1:
                    continue
            X_test.append(desc)
            y_test.append(label)
            atom_idx_test.append(atom_idx)
            idx_test.append(idx)
            comp_names.append(row[col_name])

    df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(len(X_test[0]))])
    df["label"] = y_test
    df["atom_index"] = atom_idx_test
    df["idx_name"] = idx_test
    df["comp_name"] = comp_names

    return df


def slice_pred_to_comp(lst_pred, lst_length_atom_idx):
    """Slice the list of predictions to the length of the atom indices
        making up the molecules.
    Args:
        lst_pred (list): List of predictions.
        lst_length_atom_idx (list): List of the length of the atom indices

    Returns:
        pred_comp (list): List of lists of predictions sliced to the length of the atom indices
    """

    lst_pred = iter(lst_pred)
    lst_pred_comp = [list(islice(lst_pred, elem)) for elem in lst_length_atom_idx]
    return lst_pred_comp


def pred_atoms_from_clas(model, df, name):
    df = df.copy()
    X_data = []

    for _, row in df.iterrows():
        for i in range(len(row.lst_atom_index)):
            X_data.append(row["descriptor_vector"][i])

    df_test = pd.DataFrame(X_data)

    # use the length of the atom index to later split the data and add the predictions to the original dataframe as a list of predictions
    lst_len_atom_index = [len(a_idx) for a_idx in df.lst_atom_index.values]
    # #predict on reaxys data
    reaxys_predictions = model.predict(df_test.to_numpy())
    reaxys_predictions = reaxys_predictions.round().astype(int)

    lst_pred_comp_clas = slice_pred_to_comp(reaxys_predictions, lst_len_atom_index)
    df[f"atom_lowest_pred_{name}"] = np.nan
    df[f"atom_lowest_pred_{name}"] = df[f"atom_lowest_pred_{name}"].astype("object")
    for idx, out in enumerate(lst_pred_comp_clas):
        df.at[idx, f"atom_lowest_pred_{name}"] = out

    y_site_true = reduce(operator.concat, df["atom_reaction"])
    y_site_pred = reduce(operator.concat, df[f"atom_lowest_pred_{name}"])
    cm = confusion_matrix(y_site_true, y_site_pred)
    acc = accuracy_score(y_site_true, y_site_pred)
    mcc = matthews_corrcoef(y_site_true, y_site_pred)

    data_clas = {"df": df, "cm": cm, "acc": acc, "mcc": mcc}

    return data_clas


def pred_atoms_from_regression_v2(model, df, name):
    df = df.copy()
    X_data = []

    for _, row in df.iterrows():
        for i in range(len(row.lst_atom_index)):
            X_data.append(row["descriptor_vector"][i])

    df_test = pd.DataFrame(X_data)

    # use the length of the atom index to later split the data and add the predictions to the original dataframe as a list of predictions
    lst_len_atom_index = [len(a_idx) for a_idx in df.lst_atom_index.values]
    # #predict on reaxys data
    reaxys_predictions = model.predict(df_test.to_numpy())

    lst_pred_comp_reg = slice_pred_to_comp(reaxys_predictions, lst_len_atom_index)
    df[f"lst_pka_pred_{name}"] = np.nan
    df[f"lst_pka_pred_{name}"] = df[f"lst_pka_pred_{name}"].astype("object")
    for idx, out in enumerate(lst_pred_comp_reg):
        df.at[idx, f"lst_pka_pred_{name}"] = out

    # get the minimum predicted pka value and one hot encode
    df[f"pka_min_pred_{name}"] = df[f"lst_pka_pred_{name}"].apply(lambda x: min(x))
    df[f"atom_lowest_pred_{name}"] = df.apply(
        lambda row: [
            1 if i == row[f"pka_min_pred_{name}"] else 0
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}1"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 1
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}2"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 2
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )

    y_site_true = reduce(operator.concat, df["atom_reaction"])
    y_site_pred = reduce(operator.concat, df[f"atom_lowest_pred_{name}"])
    y_site_pred1 = reduce(operator.concat, df[f"atom_lowest_pred_{name}1"])
    y_site_pred2 = reduce(operator.concat, df[f"atom_lowest_pred_{name}2"])

    cm = confusion_matrix(y_site_true, y_site_pred)
    cm_plus1 = confusion_matrix(y_site_true, y_site_pred1)
    cm_plus2 = confusion_matrix(y_site_true, y_site_pred2)

    acc = accuracy_score(y_site_true, y_site_pred)
    mcc = matthews_corrcoef(y_site_true, y_site_pred)
    eval_params = calc_mcc_from_cm(cm)

    acc_plus1 = accuracy_score(y_site_true, y_site_pred1)
    mcc_plus1 = matthews_corrcoef(y_site_true, y_site_pred1)
    eval_params1 = calc_mcc_from_cm(cm_plus1)
    acc_plus2 = accuracy_score(y_site_true, y_site_pred2)
    mcc_plus2 = matthews_corrcoef(y_site_true, y_site_pred2)
    eval_params2 = calc_mcc_from_cm(cm_plus2)

    data_reg = {
        "df": df,
        "cm": cm,
        "acc": acc,
        "mcc": mcc,
        "cm_plus1": cm_plus1,
        "acc_plus1": acc_plus1,
        "mcc_plus1": mcc_plus1,
        "cm_plus2": cm_plus2,
        "acc_plus2": acc_plus2,
        "mcc_plus2": mcc_plus2,
    }

    return data_reg


def pred_atoms_from_regression_v3(model, df, name):

    df = df.copy()
    X_data = []

    for _, row in df.iterrows():
        for i in range(len(row.atom_index)):
            X_data.append(row["descriptor_vector_conf20_shells6"][i])

    df_test = pd.DataFrame(X_data)

    # use the length of the atom index to later split the data and add the predictions to the original dataframe as a list of predictions
    lst_len_atom_index = [len(a_idx) for a_idx in df.atom_index.values]
    # #predict on reaxys data
    reaxys_predictions = model.predict(df_test.to_numpy())

    lst_pred_comp_reg = slice_pred_to_comp(reaxys_predictions, lst_len_atom_index)
    df[f"lst_pka_pred_{name}"] = np.nan
    df[f"lst_pka_pred_{name}"] = df[f"lst_pka_pred_{name}"].astype("object")
    for idx, out in enumerate(lst_pred_comp_reg):
        df.at[idx, f"lst_pka_pred_{name}"] = out

    # get the minimum predicted pka value and one hot encode
    df[f"pka_min_pred_{name}"] = df[f"lst_pka_pred_{name}"].apply(lambda x: min(x))
    df[f"atom_lowest_pred_{name}"] = df.apply(
        lambda row: [
            1 if i == row[f"pka_min_pred_{name}"] else 0
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}1"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 1
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}2"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 2
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )

    y_site_true = reduce(operator.concat, df["atom_lowest_lfer"])
    y_site_pred = reduce(operator.concat, df[f"atom_lowest_pred_{name}"])
    y_site_pred1 = reduce(operator.concat, df[f"atom_lowest_pred_{name}1"])
    y_site_pred2 = reduce(operator.concat, df[f"atom_lowest_pred_{name}2"])

    cm = confusion_matrix(y_site_true, y_site_pred)
    cm_plus1 = confusion_matrix(y_site_true, y_site_pred1)
    cm_plus2 = confusion_matrix(y_site_true, y_site_pred2)

    acc = accuracy_score(y_site_true, y_site_pred)
    mcc = matthews_corrcoef(y_site_true, y_site_pred)
    # eval_params = calc_mcc_from_cm(cm)

    acc_plus1 = accuracy_score(y_site_true, y_site_pred1)
    mcc_plus1 = matthews_corrcoef(y_site_true, y_site_pred1)
    # eval_params1 = calc_mcc_from_cm(cm_plus1)
    acc_plus2 = accuracy_score(y_site_true, y_site_pred2)
    mcc_plus2 = matthews_corrcoef(y_site_true, y_site_pred2)
    # eval_params2 = calc_mcc_from_cm(cm_plus2)

    data_reg = {
        "df": df,
        "cm": cm,
        "acc": acc,
        "mcc": mcc,
        "cm_plus1": cm_plus1,
        "acc_plus1": acc_plus1,
        "mcc_plus1": mcc_plus1,
        "cm_plus2": cm_plus2,
        "acc_plus2": acc_plus2,
        "mcc_plus2": mcc_plus2,
    }

    return data_reg


def pred_atoms_from_regression_v4(model, df, name, desc_vect_name="descriptor_vector"):

    df = df.copy()
    X_data = []

    for _, row in df.iterrows():
        for i in range(len(row[desc_vect_name])):
            X_data.append(row[desc_vect_name][i])

    df_test = pd.DataFrame(X_data)

    # use the length of the atom index to later split the data and add the predictions to the original dataframe as a list of predictions
    lst_len_atom_index = [len(a_idx) for a_idx in df[desc_vect_name].values]
    # #predict on reaxys data
    reaxys_predictions = model.predict(df_test.to_numpy())

    lst_pred_comp_reg = slice_pred_to_comp(reaxys_predictions, lst_len_atom_index)
    df[f"lst_pka_pred_{name}"] = np.nan
    df[f"lst_pka_pred_{name}"] = df[f"lst_pka_pred_{name}"].astype("object")
    for idx, out in enumerate(lst_pred_comp_reg):
        df.at[idx, f"lst_pka_pred_{name}"] = out

    # get the minimum predicted pka value and one hot encode
    df[f"pka_min_pred_{name}"] = df[f"lst_pka_pred_{name}"].apply(lambda x: min(x))
    df[f"atomsite_min_pred_{name}"] = df.apply(
        lambda row: row["lst_atomsite_deprot"][
            row[f"lst_pka_pred_{name}"].index(min(row[f"lst_pka_pred_{name}"]))
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}"] = df.apply(
        lambda row: [
            1 if i == row[f"pka_min_pred_{name}"] else 0
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}1"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 1
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}2"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 2
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )

    y_site_true = reduce(operator.concat, df["atom_lowest_lfer"])
    y_site_true1 = reduce(operator.concat, df["atom_lowest_lfer_1"])
    y_site_true2 = reduce(operator.concat, df["atom_lowest_lfer_2"])
    y_site_pred = reduce(operator.concat, df[f"atom_lowest_pred_{name}"])
    y_site_pred1 = reduce(operator.concat, df[f"atom_lowest_pred_{name}1"])
    y_site_pred2 = reduce(operator.concat, df[f"atom_lowest_pred_{name}2"])

    y_site_true, y_site_pred = zip(
        *[
            (y_true, y_pred)
            for y_true, y_pred in zip(y_site_true, y_site_pred)
            if y_true != -1
        ]
    )
    y_site_true1, y_site_pred1 = zip(
        *[
            (y_true, y_pred)
            for y_true, y_pred in zip(y_site_true1, y_site_pred1)
            if y_true != -1
        ]
    )
    y_site_true2, y_site_pred2 = zip(
        *[
            (y_true, y_pred)
            for y_true, y_pred in zip(y_site_true2, y_site_pred2)
            if y_true != -1
        ]
    )

    cm = confusion_matrix(y_site_true, y_site_pred)
    cm_plus1 = confusion_matrix(y_site_true1, y_site_pred1)
    cm_plus2 = confusion_matrix(y_site_true2, y_site_pred2)

    acc = accuracy_score(y_site_true, y_site_pred)
    mcc = matthews_corrcoef(y_site_true, y_site_pred)
    # eval_params = calc_mcc_from_cm(cm)

    acc_plus1 = accuracy_score(y_site_true1, y_site_pred1)
    mcc_plus1 = matthews_corrcoef(y_site_true1, y_site_pred1)
    # eval_params1 = calc_mcc_from_cm(cm_plus1)
    acc_plus2 = accuracy_score(y_site_true2, y_site_pred2)
    mcc_plus2 = matthews_corrcoef(y_site_true2, y_site_pred2)
    # eval_params2 = calc_mcc_from_cm(cm_plus2)

    data_reg = {
        "df": df,
        "cm": cm,
        "acc": acc,
        "mcc": mcc,
        "cm_plus1": cm_plus1,
        "acc_plus1": acc_plus1,
        "mcc_plus1": mcc_plus1,
        "cm_plus2": cm_plus2,
        "acc_plus2": acc_plus2,
        "mcc_plus2": mcc_plus2,
    }

    return data_reg


def pred_atoms_from_regression_to_thf(model, df, name, dmso_to_thf):
    df = df.copy()
    X_data = []

    for _, row in df.iterrows():
        for i in range(len(row.lst_atom_index)):
            X_data.append(row["descriptor_vector"][i])

    df_test = pd.DataFrame(X_data)

    # use the length of the atom index to later split the data and add the predictions to the original dataframe as a list of predictions
    lst_len_atom_index = [len(a_idx) for a_idx in df.lst_atom_index.values]
    # #predict on reaxys data
    reaxys_predictions = model.predict(df_test.to_numpy())

    if dmso_to_thf:
        reaxys_predictions = [
            pka_dmso_to_pka_thf(pka, reverse=False) for pka in reaxys_predictions
        ]

    lst_pred_comp_reg = slice_pred_to_comp(reaxys_predictions, lst_len_atom_index)
    df[f"lst_pka_pred_{name}"] = np.nan
    df[f"lst_pka_pred_{name}"] = df[f"lst_pka_pred_{name}"].astype("object")
    for idx, out in enumerate(lst_pred_comp_reg):
        df.at[idx, f"lst_pka_pred_{name}"] = out

    # get the minimum predicted pka value and one hot encode
    df[f"pka_min_pred_{name}"] = df[f"lst_pka_pred_{name}"].apply(lambda x: min(x))
    df[f"atom_lowest_pred_{name}"] = df.apply(
        lambda row: [
            1 if i == row[f"pka_min_pred_{name}"] else 0
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}1"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 1
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}1_5"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 1.5
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )
    df[f"atom_lowest_pred_{name}2"] = df.apply(
        lambda row: [
            (
                1
                if abs(i - row[f"pka_min_pred_{name}"]) <= 2
                or i == row[f"pka_min_pred_{name}"]
                else 0
            )
            for i in row[f"lst_pka_pred_{name}"]
        ],
        axis=1,
    )

    y_site_true = reduce(operator.concat, df["atom_reaction"])
    y_site_pred = reduce(operator.concat, df[f"atom_lowest_pred_{name}"])
    y_site_pred1 = reduce(operator.concat, df[f"atom_lowest_pred_{name}1"])
    y_site_pred2 = reduce(operator.concat, df[f"atom_lowest_pred_{name}2"])

    cm = confusion_matrix(y_site_true, y_site_pred)
    cm_plus1 = confusion_matrix(y_site_true, y_site_pred1)
    cm_plus2 = confusion_matrix(y_site_true, y_site_pred2)

    acc = accuracy_score(y_site_true, y_site_pred)
    mcc = matthews_corrcoef(y_site_true, y_site_pred)
    eval_params = calc_mcc_from_cm(cm)

    acc_plus1 = accuracy_score(y_site_true, y_site_pred1)
    mcc_plus1 = matthews_corrcoef(y_site_true, y_site_pred1)
    eval_params1 = calc_mcc_from_cm(cm_plus1)
    acc_plus2 = accuracy_score(y_site_true, y_site_pred2)
    mcc_plus2 = matthews_corrcoef(y_site_true, y_site_pred2)
    eval_params2 = calc_mcc_from_cm(cm_plus2)

    data_reg = {
        "df": df,
        "cm": cm,
        "acc": acc,
        "mcc": mcc,
        "cm_plus1": cm_plus1,
        "acc_plus1": acc_plus1,
        "mcc_plus1": mcc_plus1,
        "cm_plus2": cm_plus2,
        "acc_plus2": acc_plus2,
        "mcc_plus2": mcc_plus2,
    }

    return data_reg


# ----------------- ML METRICS -----------------


def get_mean_scores_reg(lst_scores: list, lst_model: list):
    # Extract 'train' and 'valid' items
    train_items = [d["train"] for d in lst_scores]
    valid_items = [d["valid"] for d in lst_scores]

    # Calculate the mean of 'rmse' and 'l1' for 'train'
    train_rmse_mean = np.mean([d["rmse"] for d in train_items])
    train_rmse_std = np.std([d["rmse"] for d in train_items], ddof=1)
    train_l1_mean = np.mean([d["l1"] for d in train_items])
    train_l1_std = np.std([d["l1"] for d in train_items], ddof=1)

    # Calculate the mean of 'rmse' and 'l1' for 'valid'
    valid_rmse_mean = np.mean([d["rmse"] for d in valid_items])
    valid_rmse_std = np.std([d["rmse"] for d in valid_items], ddof=1)
    valid_l1_mean = np.mean([d["l1"] for d in valid_items])
    valid_l1_std = np.std([d["l1"] for d in valid_items], ddof=1)

    best_fold = np.argmin([d["rmse"] for d in valid_items])
    best_model = lst_model[best_fold]

    return (
        best_fold,
        best_model,
        train_items,
        valid_items,
        train_rmse_mean,
        train_rmse_std,
        train_l1_mean,
        train_l1_std,
        valid_rmse_mean,
        valid_rmse_std,
        valid_l1_mean,
        valid_l1_std,
    )


def get_mean_scores_clas(lst_scores: list, lst_model: list):
    # Extract 'train' and 'valid' items
    train_items = [d["train"] for d in lst_scores]
    valid_items = [d["valid"] for d in lst_scores]

    # Calculate the mean of 'rmse' and 'l1' for 'train'
    train_logloss_mean = np.mean([d["binary_logloss"] for d in train_items])
    train_auc_mean = np.mean([d["auc"] for d in train_items])

    # Calculate the mean of 'rmse' and 'l1' for 'valid'
    valid_logloss_mean = np.mean([d["binary_logloss"] for d in valid_items])
    valid_auc_mean = np.mean([d["auc"] for d in valid_items])

    best_fold = np.argmin([d["binary_logloss"] for d in valid_items])

    best_model = lst_model[best_fold]

    return (
        best_fold,
        best_model,
        train_items,
        valid_items,
        train_logloss_mean,
        train_auc_mean,
        valid_logloss_mean,
        valid_auc_mean,
    )


def calc_mcc_from_cm(cm):
    print(f"ACC: {sum(cm.diagonal()/cm.sum()):3}")
    # if cm.shape == (1, 1):  # Check if the confusion matrix is 1x1
    #     print("Confusion matrix is 1x1, indicating a null model with no positive predictions.")
    #     print("MCC cannot be calculated for this case.")
    #     TP = 0
    #     TN = cm[0, 0]
    #     FP = 0
    #     FN = 0
    #     MCC = 0
    #     evaluation_params = {
    #         'cm': cm,
    #         'TP': 0,
    #         'TN': cm[0, 0],
    #         'FP': 0,
    #         'FN': 0,
    #         'ACC': cm[0, 0] / cm.sum(),
    #         'MCC': MCC,  # Return NaN for MCC
    #         'PPV': TP/(TP+FP),  # Return NaN for PPV
    #         'TPR': TP/(TP+FN),  # Return NaN for TPR
    #         'TNR': TN/(TN+FP),  # Return NaN for TNP
    #         'NPV': TN/(TN+FN)   # Return NaN for NPV
    #     }
    #     return evaluation_params

    # Calculate the MCC
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # print(cm)
    # print(f"MCC: {MCC:.3f}")
    # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} ")
    # print(f"PPV: {TP/(TP+FP):.3f}")
    # print(f"TPR: {TP/(TP+FN):.3f}")
    # print(f"TNR: {TN/(TN+FP):.3f}")
    # print(f"NPV: {TN/(TN+FN):.3f}")

    evaluation_params = {
        "cm": cm,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "ACC": sum(cm.diagonal() / cm.sum()),
        "MCC": MCC,
        "PPV": TP / (TP + FP),
        "TPR": TP / (TP + FN),
        "TNP": TN / (TN + FP),
        "NPV": TN / (TN + FN),
    }

    return evaluation_params


def compute_scores_regression(y_true, y_pred):
    spearman, _ = spearmanr(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    return {
        "r\u00b2": r2_score(y_true, y_pred),
        "r": pearson,
        r"$\rho$": spearman,
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
    }


# ----------------- MISC -----------------


def explode_dataframe(df, query_name, cols):
    df_query = df.query(f'names == "{query_name}"')

    # Create a new DataFrame with the same index as df for each column
    data = {col: pd.Series(df_query[col].values[0]) for col in cols}

    # Concatenate the series together into a DataFrame
    df_exploded = pd.DataFrame(data)

    # Reset the index
    df_exploded = df_exploded.reset_index(drop=True)  # .set_index(['names', 'pka_exp'])

    return df_exploded


def pka_dmso_to_pka_thf(pka: float, reverse=False) -> float:
    pka_thf = -0.963 + 1.046 * pka
    if reverse:
        pka_dmso = (pka_thf + 0.963) / 1.046
        return pka_dmso

    return pka_thf


def find_diff_hyb(df):
    """
    This function finds the difference in hybridization between the neutral and deprotonated molecule.
    Returns a tuple of:
    - the idx of the dataframe
    - idx for the deprotonated smile where hyb is different from the neutral molecule in the list of smiles
    - atom map number
    """
    # find difference in hybridization
    lst_diff_hybrid = []
    lst_same_hybrid = []
    for idx, row in df.iterrows():
        # print(idx)
        react_mol_neutral = Chem.MolFromSmiles(row["ref_mol_smiles_map"])
        for deprot_idx, deprot_smi in enumerate(row["lst_smiles_deprot_map"]):
            deprot_mol = Chem.MolFromSmiles(deprot_smi)
            for atom_idx, atom in enumerate(deprot_mol.GetAtoms()):
                charge = atom.GetFormalCharge()
                atom_mapnum = atom.GetAtomMapNum()
                if charge == -1 and atom.GetSymbol() == "C":
                    for a_idx, a in enumerate(react_mol_neutral.GetAtoms()):
                        # if the hybridization between the neutral molecule and the deprotonated molecule is not the same
                        if (
                            a.GetAtomMapNum() == atom_mapnum
                            and a.GetSymbol() == "C"
                            and a.GetHybridization() != atom.GetHybridization()
                        ):
                            lst_diff_hybrid.append((idx, deprot_idx, atom_mapnum))
                        if (
                            a.GetAtomMapNum() == atom_mapnum
                            and a.GetSymbol() == "C"
                            and a.GetHybridization() == atom.GetHybridization()
                        ):
                            lst_same_hybrid.append((idx, deprot_idx, atom_mapnum))
    return lst_diff_hybrid, lst_same_hybrid


def find_diff_hyb_min_site(df):
    """
    This function finds the difference in hybridization between the neutral and deprotonated molecule.
    Returns a tuple of:
    - the idx of the dataframe
    - idx for the deprotonated smile where hyb is different from the neutral molecule in the list of smiles
    - atom map number
    """
    # find difference in hybridization
    lst_diff_hybrid = []
    lst_same_hybrid = []
    for idx, row in df.iterrows():
        react_mol_neutral = Chem.MolFromSmiles(row["ref_mol_smiles_map"])
        idx_min_pka = np.argmin(row.lst_pka_lfer)
        atom_site_min = row.lst_atomsite[idx_min_pka]
        deprot_mol = Chem.MolFromSmiles(row["lst_smiles_deprot_map"][idx_min_pka])
        for atom_idx, atom in enumerate(deprot_mol.GetAtoms()):
            deprot_hyb = atom.GetHybridization()
            charge = atom.GetFormalCharge()
            atom_mapnum = atom.GetAtomMapNum()
            if charge == -1 and atom.GetSymbol() == "C":
                for a_idx, a in enumerate(react_mol_neutral.GetAtoms()):
                    neutral_hyb = a.GetHybridization()
                    # if the hybridization between the neutral molecule and the deprotonated molecule is not the same
                    if (
                        a.GetAtomMapNum() == atom_mapnum
                        and a.GetSymbol() == "C"
                        and a.GetHybridization() != atom.GetHybridization()
                    ):
                        lst_diff_hybrid.append(
                            (idx, atom_site_min, atom_mapnum, neutral_hyb, deprot_hyb)
                        )
                    if (
                        a.GetAtomMapNum() == atom_mapnum
                        and a.GetSymbol() == "C"
                        and a.GetHybridization() == atom.GetHybridization()
                    ):
                        lst_same_hybrid.append(
                            (idx, atom_site_min, atom_mapnum, neutral_hyb, deprot_hyb)
                        )
    return lst_diff_hybrid, lst_same_hybrid


if __name__ == "__main__":
    from ml_optimizer import split_data

    dataset = pd.read_pickle("ML_BordwellCH_2.pkl")
    dataset.drop(
        columns=[
            "group",
            "comment",
            "mol_neutral",
            "gfn_method",
            "solvent_model",
            "solvent_name",
            "amine",
            "alcohol",
            "amide",
            "pka_min_idx_sp",
            "pka_min_2_idx_xtb",
            "pka_min_2_idx_sp",
            "convergence",
            "termination",
            "lst_idx_err_term_conv",
        ],
        inplace=True,
    )

    df_prep_all, df_regression, df_unseen_regression, lst_len_atom_index = prepare_data(
        df=dataset,
        descriptor_col="descriptor_vector_conf20",
        target_col="lst_pka_sp_lfer",
    )

    X_train, X_test, y_train, y_test = split_data(
        df=df_regression, test_size=0.2, shuffle=True, random_state=42
    )
    with open(
        "/Users/borup/Nextcloud/Education/phd_theoretical_chemistry/project_CH/prelim_data/pkalc_ml/final_model_optuna.txt",
        "r",
    ) as f:
        model_str = f.read()

    # Create booster object
    booster = lgb.Booster(model_str=model_str)
    predictions_from_optuna = booster.predict(np.array(X_test))

    plot_multiple_subplots(
        2,
        np.array(y_test),
        [predictions_from_optuna, predictions_from_optuna],
        ["Model 1", "Model 2"],
        save_fig=False,
        fig_name="predictions_optuna.png",
    )

    print(y_test.index)
    print(df_regression.iloc[y_test.index].label.values)
