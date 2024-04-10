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
reports_exp_val_dir = home_directory.joinpath("reports/val_exp_radicals")


# ---------------------------------------
#               OPT FREQ
# ---------------------------------------

df_processsed_optfreq_r2scan = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_bde_optfreq_r2scan_3c_exp_val"),
    prelim_path=Path(
        qm_calculations_dir, "df_prelim_calc_bde_optfreq_r2scan_3c_exp_val_20240408.pkl"
    ),
)
df_processsed_optfreq_r2scan_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(
        qm_calculations_dir, "submit_bde_optfreq_r2scan_3c_exp_val_toluenes"
    ),
    prelim_path=Path(
        qm_calculations_dir,
        "df_prelim_calc_bde_optfreq_r2scan_3c_exp_val_toluenes_20240408.pkl",
    ),
)

df_processsed_optfreq_r2scan["names_num"] = (
    df_processsed_optfreq_r2scan["names"].str.extract("(\d+)").astype(int)
)
df_processsed_optfreq_r2scan = df_processsed_optfreq_r2scan.sort_values("names_num")
df_processsed_optfreq_r2scan.reset_index(drop=True, inplace=True)

df_processsed_optfreq_r2scan_toluenes_MeCN["names_num"] = (
    df_processsed_optfreq_r2scan_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_optfreq_r2scan_toluenes_MeCN = (
    df_processsed_optfreq_r2scan_toluenes_MeCN.sort_values("names_num")
)
df_processsed_optfreq_r2scan_toluenes_MeCN.reset_index(drop=True, inplace=True)

# replace old calculations with correct ones
df_processsed_optfreq_r2scan_all = df_processsed_optfreq_r2scan.copy()
indices_to_replace = df_processsed_optfreq_r2scan_all[
    df_processsed_optfreq_r2scan_all.names.isin(
        df_processsed_optfreq_r2scan_toluenes_MeCN.names
    )
].index

df_processsed_optfreq_r2scan_toluenes_MeCN.index = indices_to_replace
df_processsed_optfreq_r2scan_all.loc[indices_to_replace] = (
    df_processsed_optfreq_r2scan_toluenes_MeCN
)

# ---------------------------------------
#                  SP
# ---------------------------------------
# r2scan-3c

df_processsed_sp_r2scan = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_bde_sp_r2scan_3c_exp_val"),
    prelim_path=Path(
        qm_calculations_dir, "df_prelim_calc_bde_sp_r2scan_3c_exp_val_20240408.pkl"
    ),
)

df_processsed_sp_r2scan_toluenes_MeCN = process_submitted_files_halator(
    path_submitit=Path(qm_calculations_dir, "submit_bde_sp_r2scan_3c_exp_val_toluenes"),
    prelim_path=Path(
        qm_calculations_dir,
        "df_prelim_calc_bde_sp_r2scan_3c_exp_val_toluenes_20240408.pkl",
    ),
)

df_processsed_sp_r2scan["names_num"] = (
    df_processsed_sp_r2scan["names"].str.extract("(\d+)").astype(int)
)
df_processsed_sp_r2scan = df_processsed_sp_r2scan.sort_values("names_num")
df_processsed_sp_r2scan.reset_index(drop=True, inplace=True)
df_processsed_sp_r2scan_toluenes_MeCN["names_num"] = (
    df_processsed_sp_r2scan_toluenes_MeCN["names"].str.extract("(\d+)").astype(int)
)
df_processsed_sp_r2scan_toluenes_MeCN = (
    df_processsed_sp_r2scan_toluenes_MeCN.sort_values("names_num")
)
df_processsed_sp_r2scan_toluenes_MeCN.reset_index(drop=True, inplace=True)

df_processsed_sp_r2scan_all = df_processsed_sp_r2scan.copy()
indices_to_replace = df_processsed_sp_r2scan_all[
    df_processsed_sp_r2scan_all.names.isin(df_processsed_sp_r2scan_toluenes_MeCN.names)
].index

df_processsed_sp_r2scan_toluenes_MeCN.index = indices_to_replace
df_processsed_sp_r2scan_all.loc[indices_to_replace] = (
    df_processsed_sp_r2scan_toluenes_MeCN
)

# ---------------------------------------
#                  Pickling
# ---------------------------------------

df_processsed_optfreq_r2scan_all.to_pickle(
    Path(qm_calculations_dir, "df_processed_calc_bde_optfreq_r2scan_3c_exp_val.pkl")
)
df_processsed_sp_r2scan_all.to_pickle(
    Path(qm_calculations_dir, "df_processed_calc_bde_sp_r2scan_3c_exp_val.pkl")
)

# ---------------------------------------
#      PLOT QM pred HA vs QM pred BDE
# ---------------------------------------

# load processed data
df_HA_optfreq_camb3lypd4 = pd.read_pickle(
    qm_calculations_dir / "df_processed_calc_HA_optfreq_camb3lypd4_exp_val.pkl"
)
df_HA_optfreq_r2scan_3c = pd.read_pickle(
    qm_calculations_dir / "df_processed_calc_HA_optfreq_r2scan_3c_exp_val.pkl"
)
df_HA_sp_camb3lypd4 = pd.read_pickle(
    qm_calculations_dir / "df_processed_calc_HA_sp_camb3lypd4_exp_val.pkl"
)
df_HA_sp_r2scan_3c = pd.read_pickle(
    qm_calculations_dir / "df_processed_calc_HA_sp_r2scan_3c_exp_val.pkl"
)


print("------------------------")
print("xTB")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_xtb, intercept_xtb = plot_single_subplot_delta_g_halator(
    x=df_HA_sp_r2scan_3c[df_HA_sp_r2scan_3c.e_rel_min_xtb != float("inf")][
        "e_rel_min_xtb"
    ].values,
    y=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_xtb != float("inf")
    ]["e_rel_min_xtb"].values,
    title="",
    y_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ BDE [kcal/mol]",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ HA [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/xtb_HA_vs_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)
print(f"coef: {coef_xtb}, intercept: {intercept_xtb}")


print("------------------------")
print("R2SCAN-3c SP")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_r2scan_sp, intercept_r2scan_sp = plot_single_subplot_delta_g_halator(
    x=df_HA_sp_r2scan_3c[df_HA_sp_r2scan_3c.e_rel_min_dft != float("inf")][
        "e_rel_min_dft"
    ].values,
    y=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    title="",
    y_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ BDE [kcal/mol]",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ HA [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_sp_HA_vs_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_sp}, intercept: {intercept_r2scan_sp}")

print("------------------------")
print("R2SCAN-3c OPTFREQ")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_r2scan_optfreq, intercept_r2scan_optfreq = plot_single_subplot_delta_g_halator(
    x=df_HA_optfreq_r2scan_3c[df_HA_optfreq_r2scan_3c.e_rel_min_dft != float("inf")][
        "e_rel_min_dft"
    ].values,
    y=df_processsed_optfreq_r2scan_all[
        df_processsed_optfreq_r2scan_all.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    title="",
    y_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ BDE [kcal/mol]",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ HA [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_optfreq_HA_vs_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_optfreq}, intercept: {intercept_r2scan_optfreq}")


# ---------------------------------------
#      PLOT QM pred BDE vs HA exp
# ---------------------------------------


print("------------------------")
print("xTB")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_xtb, intercept_xtb = plot_single_subplot_delta_g_halator(
    x=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_xtb != float("inf")
    ]["e_rel_min_xtb"].values,
    y=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_xtb != float("inf")
    ]["HA_exp"].values,
    title="",
    y_label="Experimental HA",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/xtb_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)
print(f"coef: {coef_xtb}, intercept: {intercept_xtb}")

print("------------------------")
print("R2SCAN-3c SP")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_r2scan_sp, intercept_r2scan_sp = plot_single_subplot_delta_g_halator(
    x=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_sp_r2scan_all[
        df_processsed_sp_r2scan_all.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    y_label="Experimental HA",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_sp_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_sp}, intercept: {intercept_r2scan_sp}")

print("------------------------")
print("R2SCAN-3c OPTFREQ")
fig_size = (6, 6)
f = plt.figure(figsize=fig_size)
f, coef_r2scan_optfreq, intercept_r2scan_optfreq = plot_single_subplot_delta_g_halator(
    x=df_processsed_optfreq_r2scan_all[
        df_processsed_optfreq_r2scan_all.e_rel_min_dft != float("inf")
    ]["e_rel_min_dft"].values,
    y=df_processsed_optfreq_r2scan_all[
        df_processsed_optfreq_r2scan_all.e_rel_min_dft != float("inf")
    ]["HA_exp"].values,
    title="",
    y_label="Experimental HA",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]",
    save_fig=True,
    save_format="pdf",
    fig_name=f"{str(reports_exp_val_dir)}/r2scan_optfreq_BDE_exp_val",
    textstr=None,
    outliers=False,
    residual=8,
    fig=f,
)

print(f"coef: {coef_r2scan_optfreq}, intercept: {intercept_r2scan_optfreq}")
