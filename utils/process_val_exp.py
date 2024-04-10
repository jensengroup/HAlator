import sys
import os

from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint
import joblib
from operator import concat
from functools import reduce
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

fm.fontManager.addfont("/groups/kemi/borup/HAlator/utils/Helvetica.ttf")
fm.findfont("Helvetica", fontext="ttf", rebuild_if_missing=False)


current_directory = Path.cwd()
# Get the parent directory until you reach 'HAlator'
home_directory = current_directory
while home_directory.name != "HAlator":
    if home_directory == home_directory.parent:  # We've reached the root directory
        raise Exception("HAlator directory not found")
    home_directory = home_directory.parent


sys.path.append(str(home_directory / "qm_halator"))

from etl import process_submitted_files_halator, pred_HA
from visualize import (
    plot_multiple_subplots_delta_g_halator,
    plot_single_subplot_delta_g_halator,
    plot_single_subplot_calc_pred_halator,
    create_benchmark_hist,
)

qm_calculations_dir = home_directory.joinpath("data/qm_calculations")
reports_exp_val_dir = home_directory.joinpath("reports/val_exp")


# ---------------------------------------
#               OPT FREQ
# ---------------------------------------


df_processsed_camb3lypd4 = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir / "submit_HA_optfreq_camb3lypd4_exp_val"),
    prelim_path=Path(
        qm_calculations_dir
        / "df_prelim_calc_HA_optfreq_camb3lypd4_exp_val_20240325.pkl"
    ),
)
df_processsed_camb3lypd4_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(
        qm_calculations_dir / "submit_HA_optfreq_camb3lypd4_exp_val_toluenes2"
    ),
    prelim_path=Path(
        qm_calculations_dir
        / "df_prelim_calc_HA_optfreq_camb3lypd4_exp_val_toluenes2_20240325.pkl"
    ),
)
df_processsed_camb3lypd4_fluorenes = process_submitted_files_halator(
    path_submitit=Path(
        qm_calculations_dir / "submit_HA_optfreq_camb3lypd4_exp_val_fluorenes"
    ),
    prelim_path=Path(
        qm_calculations_dir
        / "df_prelim_calc_HA_optfreq_camb3lypd4_exp_val_fluorenes_20240325.pkl"
    ),
)
df_processsed_camb3lypd4_val26 = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir / "submit_HA_optfreq_camb3lypd4_exp_val26"),
    prelim_path=Path(
        qm_calculations_dir
        / "df_prelim_calc_HA_optfreq_camb3lypd4_exp_val26_20240326.pkl"
    ),
)

df_processsed_camb3lypd4_rerun = process_submitted_files_halator(
    path_submitit=Path(
        qm_calculations_dir, "submit_HA_optfreq_camb3lypd4_exp_val_rerun"
    ),
    prelim_path=Path(
        qm_calculations_dir,
        "df_prelim_calc_HA_optfreq_camb3lypd4_exp_val_rerun_20240327.pkl",
    ),
)

df_processsed_camb3lypd4["names_num"] = (
    df_processsed_camb3lypd4["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4 = df_processsed_camb3lypd4.sort_values("names_num")
df_processsed_camb3lypd4.reset_index(drop=True, inplace=True)
df_processsed_camb3lypd4_toluenes_MeCN["names_num"] = (
    df_processsed_camb3lypd4_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_toluenes_MeCN = (
    df_processsed_camb3lypd4_toluenes_MeCN.sort_values("names_num")
)
df_processsed_camb3lypd4_toluenes_MeCN.reset_index(drop=True, inplace=True)
df_processsed_camb3lypd4_fluorenes["names_num"] = (
    df_processsed_camb3lypd4_fluorenes["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_fluorenes = df_processsed_camb3lypd4_fluorenes.sort_values(
    "names_num"
)
df_processsed_camb3lypd4_fluorenes.reset_index(drop=True, inplace=True)
df_processsed_camb3lypd4_val26["names_num"] = (
    df_processsed_camb3lypd4_val26["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_val26 = df_processsed_camb3lypd4_val26.sort_values("names_num")
df_processsed_camb3lypd4_val26.reset_index(drop=True, inplace=True)

df_processsed_camb3lypd4_rerun["names_num"] = (
    df_processsed_camb3lypd4_rerun["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_rerun = df_processsed_camb3lypd4_rerun.sort_values("names_num")
df_processsed_camb3lypd4_rerun.reset_index(drop=True, inplace=True)


df_processsed_camb3lypd4 = pd.concat(
    [df_processsed_camb3lypd4, df_processsed_camb3lypd4_fluorenes]
)
df_processsed_camb3lypd4.reset_index(drop=True, inplace=True)


# # replace old calculations with correct ones
df_processsed_camb3lypd4_with_tolunes_MeCN = df_processsed_camb3lypd4.copy()

indices_to_replace = df_processsed_camb3lypd4_with_tolunes_MeCN[
    df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(
        df_processsed_camb3lypd4_toluenes_MeCN.names
    )
].index
df_processsed_camb3lypd4_toluenes_MeCN.index = indices_to_replace
df_processsed_camb3lypd4_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_camb3lypd4_toluenes_MeCN
)

indices_to_replace = df_processsed_camb3lypd4_with_tolunes_MeCN[
    df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(
        df_processsed_camb3lypd4_val26.names
    )
].index
df_processsed_camb3lypd4_val26.index = indices_to_replace
df_processsed_camb3lypd4_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_camb3lypd4_val26
)


indices_to_replace = df_processsed_camb3lypd4_with_tolunes_MeCN[
    df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(
        df_processsed_camb3lypd4_rerun.names
    )
].index
df_processsed_camb3lypd4_rerun.index = indices_to_replace
df_processsed_camb3lypd4_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_camb3lypd4_rerun
)

print(df_processsed_camb3lypd4_with_tolunes_MeCN.head(5))
print(df_processsed_camb3lypd4_toluenes_MeCN["solvent_name"].unique())
print(
    df_processsed_camb3lypd4_with_tolunes_MeCN[
        df_processsed_camb3lypd4_with_tolunes_MeCN["solvent_name"] == "Acetonitrile"
    ]
)

df_processsed_r2scan = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_HA_optfreq_r2scan_3c_exp_val"),
    prelim_path=Path(
        qm_calculations_dir, "df_prelim_calc_HA_optfreq_r2scan_3c_exp_val_20240321.pkl"
    ),
)
df_processsed_r2scan_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_optfreq_r2scan_3c_exp_val_toluenes"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_optfreq_r2scan_3c_exp_val_toluenes_20240325.pkl",
)
df_processsed_r2scan_fluorenes = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_optfreq_r2scan_3c_exp_val_fluorenes"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_optfreq_r2scan_3c_exp_val_fluorenes_20240325.pkl",
)
df_processsed_r2scan_val26 = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_optfreq_r2scan_3c_exp_val26"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_optfreq_r2scan_3c_exp_val26_20240326.pkl",
)

df_processsed_r2scan["names_num"] = (
    df_processsed_r2scan["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan = df_processsed_r2scan.sort_values("names_num")
df_processsed_r2scan.reset_index(drop=True, inplace=True)
df_processsed_r2scan_toluenes_MeCN["names_num"] = (
    df_processsed_r2scan_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan_toluenes_MeCN = df_processsed_r2scan_toluenes_MeCN.sort_values(
    "names_num"
)
df_processsed_r2scan_toluenes_MeCN.reset_index(drop=True, inplace=True)
df_processsed_r2scan_fluorenes["names_num"] = (
    df_processsed_r2scan_fluorenes["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan_fluorenes = df_processsed_r2scan_fluorenes.sort_values("names_num")
df_processsed_r2scan_fluorenes.reset_index(drop=True, inplace=True)
df_processsed_r2scan_val26["names_num"] = (
    df_processsed_r2scan_val26["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan_val26 = df_processsed_r2scan_val26.sort_values("names_num")
df_processsed_r2scan_val26.reset_index(drop=True, inplace=True)


df_processsed_r2scan = pd.concat([df_processsed_r2scan, df_processsed_r2scan_fluorenes])
df_processsed_r2scan.reset_index(drop=True, inplace=True)

# replace old calculations with correct ones
df_processsed_r2scan_with_tolunes_MeCN = df_processsed_r2scan.copy()
indices_to_replace = df_processsed_r2scan_with_tolunes_MeCN[
    df_processsed_r2scan_with_tolunes_MeCN.names.isin(
        df_processsed_r2scan_toluenes_MeCN.names
    )
].index
df_processsed_r2scan_toluenes_MeCN.index = indices_to_replace
df_processsed_r2scan_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_r2scan_toluenes_MeCN
)

indices_to_replace = df_processsed_r2scan_with_tolunes_MeCN[
    df_processsed_r2scan_with_tolunes_MeCN.names.isin(df_processsed_r2scan_val26.names)
].index
df_processsed_r2scan_val26.index = indices_to_replace
df_processsed_r2scan_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_r2scan_val26
)


# ---------------------------------------
#                  SP
# ---------------------------------------

# camb3lypd4

df_processsed_camb3lypd4_sp = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_sp_camb3lypd4_exp_val"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_sp_camb3lypd4_exp_val_20240326.pkl",
)

df_processsed_camb3lypd4_sp_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_sp_camb3lypd4_exp_val_toluenes"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_sp_r2scan_3c_exp_val_toluenes_20240327.pkl",
)

df_processsed_camb3lypd4_sp_val4 = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_sp_camb3lypd4_exp_val4"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_sp_camb3lypd4_exp_val4_20240327.pkl",
)

df_processsed_camb3lypd4_sp["names_num"] = (
    df_processsed_camb3lypd4_sp["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_sp = df_processsed_camb3lypd4_sp.sort_values("names_num")
df_processsed_camb3lypd4_sp.reset_index(drop=True, inplace=True)

df_processsed_camb3lypd4_sp_toluenes_MeCN["names_num"] = (
    df_processsed_camb3lypd4_sp_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_sp_toluenes_MeCN = (
    df_processsed_camb3lypd4_sp_toluenes_MeCN.sort_values("names_num")
)
df_processsed_camb3lypd4_sp_toluenes_MeCN.reset_index(drop=True, inplace=True)

df_processsed_camb3lypd4_sp_val4["names_num"] = (
    df_processsed_camb3lypd4_sp_val4["names"].str.extract("(\d+)").astype(int)
)
df_processsed_camb3lypd4_sp_val4 = df_processsed_camb3lypd4_sp_val4.sort_values(
    "names_num"
)
df_processsed_camb3lypd4_sp_val4.reset_index(drop=True, inplace=True)


df_processsed_camb3lypd4_sp_with_tolunes_MeCN = df_processsed_camb3lypd4_sp.copy()
indices_to_replace = df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
    df_processsed_camb3lypd4_sp_with_tolunes_MeCN.names.isin(
        df_processsed_camb3lypd4_sp_toluenes_MeCN.names
    )
].index
df_processsed_camb3lypd4_sp_toluenes_MeCN.index = indices_to_replace
df_processsed_camb3lypd4_sp_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_camb3lypd4_sp_toluenes_MeCN
)

indices_to_replace = df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
    df_processsed_camb3lypd4_sp_with_tolunes_MeCN.names.isin(
        df_processsed_camb3lypd4_sp_val4.names
    )
].index
df_processsed_camb3lypd4_sp_val4.index = indices_to_replace
df_processsed_camb3lypd4_sp_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_camb3lypd4_sp_val4
)


# r2scan-3c

df_processsed_r2scan_sp = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_sp_r2scan_3c_exp_val"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_sp_r2scan_3c_exp_val_20240326.pkl",
)

df_processsed_r2scan_sp_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(
        "/groups/kemi/borup/HAlator/data/qm_calculations/submit_HA_sp_r2scan_3c_exp_val_toluenes"
    ),
    prelim_path="/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_sp_r2scan_3c_exp_val_toluenes_20240327.pkl",
)

df_processsed_r2scan_sp["names_num"] = (
    df_processsed_r2scan_sp["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan_sp = df_processsed_r2scan_sp.sort_values("names_num")
df_processsed_r2scan_sp.reset_index(drop=True, inplace=True)
df_processsed_r2scan_sp_toluenes_MeCN["names_num"] = (
    df_processsed_r2scan_sp_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_r2scan_sp_toluenes_MeCN = (
    df_processsed_r2scan_sp_toluenes_MeCN.sort_values("names_num")
)
df_processsed_r2scan_sp_toluenes_MeCN.reset_index(drop=True, inplace=True)


df_processsed_r2scan_sp_with_tolunes_MeCN = df_processsed_r2scan_sp.copy()
indices_to_replace = df_processsed_r2scan_sp_with_tolunes_MeCN[
    df_processsed_r2scan_sp_with_tolunes_MeCN.names.isin(
        df_processsed_r2scan_sp_toluenes_MeCN.names
    )
].index
df_processsed_r2scan_sp_toluenes_MeCN.index = indices_to_replace
df_processsed_r2scan_sp_with_tolunes_MeCN.loc[indices_to_replace] = (
    df_processsed_r2scan_sp_toluenes_MeCN
)


# M06-2X

df_processsed_sp_m062x = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_HA_sp_m062x_exp_val"),
    prelim_path=Path(
        qm_calculations_dir, "df_prelim_calc_HA_sp_m062x_exp_val_20240408.pkl"
    ),
)


df_processsed_spm062x_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_HA_sp_m062x_exp_val_toluenes"),
    prelim_path=Path(
        qm_calculations_dir, "df_prelim_calc_HA_sp_m062x_exp_val_toluenes_20240408.pkl"
    ),
)


df_processsed_sp_m062x["names_num"] = (
    df_processsed_sp_m062x["names"].str.extract("(\d+)").astype(int)
)
df_processsed_sp_m062x = df_processsed_sp_m062x.sort_values("names_num")
df_processsed_sp_m062x.reset_index(drop=True, inplace=True)
df_processsed_spm062x_toluenes_MeCN["names_num"] = (
    df_processsed_spm062x_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_spm062x_toluenes_MeCN = df_processsed_spm062x_toluenes_MeCN.sort_values(
    "names_num"
)
df_processsed_spm062x_toluenes_MeCN.reset_index(drop=True, inplace=True)


df_processsed_sp_m062x_all = df_processsed_r2scan_sp.copy()
indices_to_replace = df_processsed_sp_m062x_all[
    df_processsed_sp_m062x_all.names.isin(df_processsed_spm062x_toluenes_MeCN.names)
].index
df_processsed_spm062x_toluenes_MeCN.index = indices_to_replace
df_processsed_sp_m062x_all.loc[indices_to_replace] = df_processsed_spm062x_toluenes_MeCN


# ---------------------------------------
#                  Plot
# ---------------------------------------


fig_size = (6, 6)
print("------------------------")
print("xTB")
f = plt.figure(figsize=fig_size)
f, coef_xtb, intercept_xtb = plot_single_subplot_delta_g_halator(
    x=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
        df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_xtb != float("inf")
    ]["e_rel_min_xtb"].values,
    y=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
        df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_xtb != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/xtb_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)
print(f"coef: {coef_xtb}, intercept: {intercept_xtb}")


print("------------------------")
print("R2SCAN-3c SP")
f = plt.figure(figsize=fig_size)
f, coef_r2scan_sp, intercept_r2scan_sp = plot_single_subplot_delta_g_halator(
    x=df_processsed_r2scan_sp_with_tolunes_MeCN[
        df_processsed_r2scan_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_r2scan_sp_with_tolunes_MeCN[
        df_processsed_r2scan_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_sp_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_sp}, intercept: {intercept_r2scan_sp}")

print("------------------------")
print("CAM-B3LYP D4 SP")
f = plt.figure(figsize=fig_size)
f, coef_camb3lypd4_sp, intercept_camb3lypd4_sp = plot_single_subplot_delta_g_halator(
    x=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
        df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
        df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/camb3lypd4_sp_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)
print(f"coef: {coef_camb3lypd4_sp}, intercept: {intercept_camb3lypd4_sp}")


print("------------------------")
print("R2SCAN-3c OPTFREQ")
f = plt.figure(figsize=fig_size)
f, coef_r2scan_optfreq, intercept_r2scan_optfreq = plot_single_subplot_delta_g_halator(
    x=df_processsed_r2scan_with_tolunes_MeCN[
        df_processsed_r2scan_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_r2scan_with_tolunes_MeCN[
        df_processsed_r2scan_with_tolunes_MeCN.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_optfreq_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_optfreq}, intercept: {intercept_r2scan_optfreq}")

print("------------------------")
print("CAM-B3LYP D4 OPTFREQ")
f = plt.figure(figsize=fig_size)
f, coef_camb3lypd4_optfreq, intercept_camb3lypd4_optfreq = (
    plot_single_subplot_delta_g_halator(
        x=df_processsed_camb3lypd4_with_tolunes_MeCN[
            ~df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(["val2", "val4"])
        ]["e_rel_min_dft"].values,
        y=df_processsed_camb3lypd4_with_tolunes_MeCN[
            ~df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(["val2", "val4"])
        ]["HA_exp"].values,
        title="",
        save_fig=True,
        save_format="pdf",
        fig_name=f"{str(reports_exp_val_dir)}/camb3lypd4_optfreq_exp_val",
        textstr=None,
        outliers=False,
        residual=8,
        fig=f,
    )
)
print(f"coef: {coef_camb3lypd4_optfreq}, intercept: {intercept_camb3lypd4_optfreq}")

df_processsed_r2scan_sp_with_tolunes_MeCN = pred_HA(
    df=df_processsed_r2scan_sp_with_tolunes_MeCN,
    coef_xtb=coef_xtb,
    intercept_xtb=intercept_xtb,
    coef_dft=coef_r2scan_sp,
    intercept_dft=intercept_r2scan_sp,
)

df_processsed_camb3lypd4_sp_with_tolunes_MeCN = pred_HA(
    df=df_processsed_camb3lypd4_sp_with_tolunes_MeCN,
    coef_xtb=coef_xtb,
    intercept_xtb=intercept_xtb,
    coef_dft=coef_camb3lypd4_sp,
    intercept_dft=intercept_camb3lypd4_sp,
)

df_processsed_camb3lypd4_with_tolunes_MeCN = pred_HA(
    df=df_processsed_camb3lypd4_with_tolunes_MeCN,
    coef_xtb=coef_xtb,
    intercept_xtb=intercept_xtb,
    coef_dft=coef_camb3lypd4_optfreq,
    intercept_dft=intercept_camb3lypd4_optfreq,
)

df_processsed_r2scan_with_tolunes_MeCN = pred_HA(
    df=df_processsed_r2scan_with_tolunes_MeCN,
    coef_xtb=coef_xtb,
    intercept_xtb=intercept_xtb,
    coef_dft=coef_r2scan_optfreq,
    intercept_dft=intercept_r2scan_optfreq,
)


print("------------------------")
print("M06-2X SP")
f = plt.figure(figsize=fig_size)
f, coef_m062_sp, intercept_m062_sp = plot_single_subplot_delta_g_halator(
    x=df_processsed_sp_m062x_all[
        df_processsed_sp_m062x_all.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_sp_m062x_all[
        df_processsed_sp_m062x_all.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/m062x_sp_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_m062_sp}, intercept: {intercept_m062_sp}")

df_processsed_sp_m062x_all = pred_HA(
    df=df_processsed_sp_m062x_all,
    coef_xtb=coef_xtb,
    intercept_xtb=intercept_xtb,
    coef_dft=coef_m062_sp,
    intercept_dft=intercept_m062_sp,
)

# ---------------------------------------
#                  Pickling
# ---------------------------------------
# df_processsed_camb3lypd4_with_tolunes_MeCN.to_pickle(
#     Path(qm_calculations_dir, "df_processed_calc_HA_optfreq_camb3lypd4_exp_val.pkl")
# )
# df_processsed_camb3lypd4_sp_with_tolunes_MeCN.to_pickle(
#     Path(qm_calculations_dir, "df_processed_calc_HA_sp_camb3lypd4_exp_val.pkl")
# )

# df_processsed_r2scan_with_tolunes_MeCN.to_pickle(
#     Path(qm_calculations_dir, "df_processed_calc_HA_optfreq_r2scan_3c_exp_val.pkl")
# )
# df_processsed_r2scan_sp_with_tolunes_MeCN.to_pickle(
#     Path(qm_calculations_dir, "df_processed_calc_HA_sp_r2scan_3c_exp_val.pkl")
# )

df_processsed_sp_m062x_all.to_pickle(
    Path(qm_calculations_dir, "df_processed_calc_HA_sp_m062x_exp_val.pkl")
)


# ---------------------------------------
#                  Plot benchmark
# ---------------------------------------
# print("Plotting benchmark")
# print("------------------------")
# print("xTB")
# f = plt.figure(figsize=fig_size)
# f, coef_xtb, intercept_xtb = plot_single_subplot_calc_pred_halator(
#     x=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_xtb != float("inf")
#     ]["HA_min_qmpred_xtb"].values,
#     y=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_xtb != float("inf")
#     ]["HA_exp"].values,
#     title="",
#     save_fig=True,
#     save_format="pdf",
#     fig_name=f"{str(reports_exp_val_dir)}/xtb_pred_exp_val",
#     textstr=None,
#     outliers=False,
#     residual=8,
#     fig=f,
# )
# print(f"coef: {coef_xtb}, intercept: {intercept_xtb}")


# print("------------------------")
# print("R2SCAN-3c SP")
# f = plt.figure(figsize=fig_size)
# f, coef_r2scan_sp, intercept_r2scan_sp = plot_single_subplot_calc_pred_halator(
#     x=df_processsed_r2scan_sp_with_tolunes_MeCN[
#         df_processsed_r2scan_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_min_qmpred_dft"].values,
#     y=df_processsed_r2scan_sp_with_tolunes_MeCN[
#         df_processsed_r2scan_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_exp"].values,
#     title="",
#     save_fig=True,
#     save_format="pdf",
#     fig_name=f"{str(reports_exp_val_dir)}/r2scan_sp_pred_exp_val",
#     textstr=None,
#     outliers=False,
#     residual=8,
#     fig=f,
# )

# print(f"coef: {coef_r2scan_sp}, intercept: {intercept_r2scan_sp}")

# print("------------------------")
# print("CAM-B3LYP D4 SP")
# f = plt.figure(figsize=fig_size)
# f, coef_camb3lypd4_sp, intercept_camb3lypd4_sp = plot_single_subplot_calc_pred_halator(
#     x=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_min_qmpred_dft"].values,
#     y=df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_exp"].values,
#     title="",
#     save_fig=True,
#     save_format="pdf",
#     fig_name=f"{str(reports_exp_val_dir)}/camb3lypd4_sp_pred_exp_val",
#     textstr=None,
#     outliers=False,
#     residual=8,
#     fig=f,
# )
# print(f"coef: {coef_camb3lypd4_sp}, intercept: {intercept_camb3lypd4_sp}")


# print("------------------------")
# print("R2SCAN-3c OPTFREQ")
# f = plt.figure(figsize=fig_size)
# f, coef_r2scan_optfreq, intercept_r2scan_optfreq = (
#     plot_single_subplot_calc_pred_halator(
#         x=df_processsed_r2scan_with_tolunes_MeCN[
#             df_processsed_r2scan_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#         ]["HA_min_qmpred_dft"].values,
#         y=df_processsed_r2scan_with_tolunes_MeCN[
#             df_processsed_r2scan_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#         ]["HA_exp"].values,
#         title="",
#         save_fig=True,
#         save_format="pdf",
#         fig_name=f"{str(reports_exp_val_dir)}/r2scan_optfreq_pred_exp_val",
#         textstr=None,
#         outliers=False,
#         residual=8,
#         fig=f,
#     )
# )

# print(f"coef: {coef_r2scan_optfreq}, intercept: {intercept_r2scan_optfreq}")

# print("------------------------")
# print("CAM-B3LYP D4 OPTFREQ")
# f = plt.figure(figsize=fig_size)
# f, coef_camb3lypd4_optfreq, intercept_camb3lypd4_optfreq = (
#     plot_single_subplot_calc_pred_halator(
#         x=df_processsed_camb3lypd4_with_tolunes_MeCN[
#             ~df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(["val2", "val4"])
#         ]["HA_min_qmpred_dft"].values,
#         y=df_processsed_camb3lypd4_with_tolunes_MeCN[
#             ~df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(["val2", "val4"])
#         ]["HA_exp"].values,
#         title="",
#         save_fig=True,
#         save_format="pdf",
#         fig_name=f"{str(reports_exp_val_dir)}/camb3lypd4_optfreq_pred_exp_val",
#         textstr=None,
#         outliers=False,
#         residual=8,
#         fig=f,
#     )
# )
# print(f"coef: {coef_camb3lypd4_optfreq}, intercept: {intercept_camb3lypd4_optfreq}")

print("------------------------")
print("M06-2X SP")
f = plt.figure(figsize=fig_size)
f, coef_m062x_sp, intercept_m062x_sp = plot_single_subplot_calc_pred_halator(
    x=df_processsed_sp_m062x_all[
        df_processsed_sp_m062x_all.e_rel_min_dft != float("inf")
    ]["HA_min_qmpred_dft"].values,
    y=df_processsed_sp_m062x_all[
        df_processsed_sp_m062x_all.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/m062x_sp_pred_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_m062x_sp}, intercept: {intercept_m062x_sp}")

# lst_errors = [
#     df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_xtb != float("inf")
#     ]["HA_min_qmpred_error_xtb"].values.flatten(),
#     df_processsed_r2scan_sp_with_tolunes_MeCN[
#         df_processsed_r2scan_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_min_qmpred_error_dft"].values.flatten(),
#     df_processsed_camb3lypd4_sp_with_tolunes_MeCN[
#         df_processsed_camb3lypd4_sp_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_min_qmpred_error_dft"].values.flatten(),
#     df_processsed_r2scan_with_tolunes_MeCN[
#         df_processsed_r2scan_with_tolunes_MeCN.e_rel_min_dft != float("inf")
#     ]["HA_min_qmpred_error_dft"].values.flatten(),
#     df_processsed_camb3lypd4_with_tolunes_MeCN[
#         ~df_processsed_camb3lypd4_with_tolunes_MeCN.names.isin(["val2", "val4"])
#     ]["HA_min_qmpred_error_dft"].values.flatten(),
# ]

# print(lst_errors)

# create_benchmark_hist(
#     lst_errors=lst_errors,
#     save_fig=True,
#     save_format="pdf",
#     fig_name=f"{str(reports_exp_val_dir)}/benchmark_hist_test",
# )
