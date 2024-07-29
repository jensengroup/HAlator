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
submit_dir = qm_calculations_dir.joinpath("results_reaction_data/submit")
prelim_dir = qm_calculations_dir.joinpath("results_reaction_data/prelim")
processed_dir = qm_calculations_dir.joinpath("results_reaction_data/processed")
# reports_exp_val_dir = home_directory.joinpath("reports/dataset")


# ---------------------------------------
#               OPT FREQ
# ---------------------------------------


# data = {
#     "reaction_data": {
#         "optfreq": {
#             "calc": "calc_HA_optfreq_r2scan_3c_reaction_data",
#             "submit": "submit_HA_optfreq_r2scan_3c_reaction_data",
#             "prelim_dataframe": "df_prelim_calc_HA_optfreq_r2scan_3c_reaction_data_20240423.pkl",
#         },
#     },
# }

data = {
    "reaction_data": {
        "optfreq": {
            "calc": "calc_HA_optfreq_r2scan_3c_reaction_data_ref43",
            "submit": "submit_HA_optfreq_r2scan_3c_reaction_data_ref43",
            "prelim_dataframe": "df_prelim_calc_HA_optfreq_r2scan_3c_reaction_data_ref43_20240611.pkl",
        },
    },
}


def process_all_reaction_data(data, submit_dir, prelim_dir, optfreq=True):
    if optfreq:
        method = "optfreq"
    else:
        method = "sp"

    dfs = []  # list to hold all dataframes

    for key in data:
        df = process_submitted_files_halator(
            path_submitit=Path(submit_dir, data[key][method]["submit"]),
            prelim_path=Path(prelim_dir, data[key][method]["prelim_dataframe"]),
        )

        # df["names_num"] = df["names"].str.extract("(\d+)").astype(int)
        # df = df.sort_values("names_num")

        dfs.append(df)

    # concatenate all dataframes and reset index
    df_result = pd.concat(dfs).reset_index(drop=True)

    df_result = pred_HA(df=df_result, predetermined=f"R2SCAN_3c_{method.upper()}")

    return df_result


df_processsed_optfreq_r2scan = process_all_reaction_data(
    data=data, submit_dir=submit_dir, prelim_dir=prelim_dir, optfreq=True
)
# df_processsed_sp_r2scan = process_all(data, qm_calculations_dir, optfreq=False)


# df_processsed_optfreq_r2scan.to_pickle(
#     Path(
#         processed_dir,
#         "df_processed_HA_optfreq_r2scan_3c_reaction_data_20240607.pkl",
#     )
# )

df_processsed_optfreq_r2scan.to_pickle(
    Path(
        processed_dir,
        "df_processed_HA_optfreq_r2scan_3c_reaction_data_ref43_20240612.pkl",
    )
)
