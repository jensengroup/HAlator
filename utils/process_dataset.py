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
reports_exp_val_dir = home_directory.joinpath("reports/dataset")


# ---------------------------------------
#               OPT FREQ
# ---------------------------------------
# calc_HA_optfreq_r2scan_3c_bordwell
# calc_HA_optfreq_r2scan_3c_ibond
# calc_HA_optfreq_r2scan_3c_shen

# submit_HA_optfreq_r2scan_3c_bordwell
# submit_HA_optfreq_r2scan_3c_ibond
# submit_HA_optfreq_r2scan_3c_shen

# df_prelim_calc_HA_optfreq_r2scan_3c_bordwell_20240405.pkl
# df_prelim_calc_HA_optfreq_r2scan_3c_ibond_20240405.pkl
# df_prelim_calc_HA_optfreq_r2scan_3c_shen_20240405.pkl


# calc_HA_sp_r2scan_3c_bordwell
# calc_HA_sp_r2scan_3c_ibond
# calc_HA_sp_r2scan_3c_shen

# submit_HA_sp_r2scan_3c_bordwell
# submit_HA_sp_r2scan_3c_ibond
# submit_HA_sp_r2scan_3c_shen

# df_prelim_calc_HA_sp_r2scan_3c_bordwell_20240405.pkl
# df_prelim_calc_HA_sp_r2scan_3c_ibond_20240405.pkl
# df_prelim_calc_HA_sp_r2scan_3c_shen_20240405.pkl


data = {
    "bordwell": {
        "optfreq": {
            "calc": "calc_HA_optfreq_r2scan_3c_bordwell",
            "submit": "submit_HA_optfreq_r2scan_3c_bordwell",
            "prelim_dataframe": "df_prelim_calc_HA_optfreq_r2scan_3c_bordwell_20240405.pkl",
        },
        "sp": {
            "calc": "calc_HA_sp_r2scan_3c_bordwell",
            "submit": "submit_HA_sp_r2scan_3c_bordwell",
            "prelim_dataframe": "df_prelim_calc_HA_sp_r2scan_3c_bordwell_20240405.pkl",
        },
    },
    "ibond": {
        "optfreq": {
            "calc": "calc_HA_optfreq_r2scan_3c_ibond",
            "submit": "submit_HA_optfreq_r2scan_3c_ibond",
            "prelim_dataframe": "df_prelim_calc_HA_optfreq_r2scan_3c_ibond_20240405.pkl",
        },
        "sp": {
            "calc": "calc_HA_sp_r2scan_3c_ibond",
            "submit": "submit_HA_sp_r2scan_3c_ibond",
            "prelim_dataframe": "df_prelim_calc_HA_sp_r2scan_3c_ibond_20240405.pkl",
        },
    },
    "shen": {
        "optfreq": {
            "calc": "calc_HA_optfreq_r2scan_3c_shen",
            "submit": "submit_HA_optfreq_r2scan_3c_shen",
            "prelim_dataframe": "df_prelim_calc_HA_optfreq_r2scan_3c_shen_20240405.pkl",
        },
        "sp": {
            "calc": "calc_HA_sp_r2scan_3c_shen",
            "submit": "submit_HA_sp_r2scan_3c_shen",
            "prelim_dataframe": "df_prelim_calc_HA_sp_r2scan_3c_shen_20240405.pkl",
        },
    },
}

# df_processsed_optfreq_r2scan_bordwell = process_submitted_files_halator(
#     path_submitit=Path(qm_calculations_dir, data["bordwell"]["optfreq"]["submit"]),
#     prelim_path=Path(
#         qm_calculations_dir, data["bordwell"]["optfreq"]["prelim_dataframe"]
#     ),
# )

# df_processsed_optfreq_r2scan_bordwell["names_num"] = (
#     df_processsed_optfreq_r2scan_bordwell["names"].str.extract("(\d+)").astype(int)
# )
# df_processsed_optfreq_r2scan_bordwell = df_processsed_optfreq_r2scan_bordwell.sort_values("names_num")


# df_processsed_optfreq_r2scan_bordwell.reset_index(drop=True, inplace=True)


def process_all(data, qm_calculations_dir, optfreq=True):
    if optfreq:
        method = "optfreq"
    else:
        method = "sp"

    dfs = []  # list to hold all dataframes

    for key in data:
        df = process_submitted_files_halator(
            path_submitit=Path(qm_calculations_dir, data[key][method]["submit"]),
            prelim_path=Path(
                qm_calculations_dir, data[key][method]["prelim_dataframe"]
            ),
        )

        df["names_num"] = df["names"].str.extract("(\d+)").astype(int)
        df = df.sort_values("names_num")

        dfs.append(df)

    # concatenate all dataframes and reset index
    df_result = pd.concat(dfs).reset_index(drop=True)

    df_result = pred_HA(df=df_result, predetermined=f"R2SCAN_3c_{method.upper()}")

    return df_result


df_processsed_optfreq_r2scan = process_all(data, qm_calculations_dir, optfreq=True)
df_processsed_sp_r2scan = process_all(data, qm_calculations_dir, optfreq=False)

df_processsed_optfreq_r2scan.to_pickle(
    Path(qm_calculations_dir, "df_processed_HA_optfreq_r2scan_3c_dataset.pkl")
)

df_processsed_sp_r2scan.to_pickle(
    Path(qm_calculations_dir, "df_processed_HA_sp_r2scan_3c_dataset.pkl")
)
