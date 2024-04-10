# utility functions to Extract, Transform, and Load data

import pandas as pd
import sys
from pathlib import Path
import argparse

# sys.path.insert(0, "/lustre/hpc/kemi/borup/pKalculator/src/smi2gcs'")
# home_directory = Path.cwd()

current_directory = Path.cwd()

# Get the parent directory until you reach 'HAlator'
home_directory = current_directory
while home_directory.name != "HAlator":
    if home_directory == home_directory.parent:  # We've reached the root directory
        raise Exception("HAlator directory not found")
    home_directory = home_directory.parent

sys.path.append(str(home_directory / "qm_halator"))
sys.path.append(str(home_directory / "smi2gcs"))
# FIX THIS WHEN WORKING IN ANOTHER DIRECTORY
from DescriptorCreator.PrepAndCalcDescriptor import Generator


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Add qm calculations to the preliminary dataframe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-calc",
        "--calc_path",
        metavar="calc_path",
        help="calculation path where the calculations are stored.",
        default="data/qm_calculations/calc_test",
        type=str,
    )

    parser.add_argument(
        "-submit",
        "--submit_path",
        metavar="submit_path",
        help="submit path where log files are stored.",
        default="data/qm_calculations/submit_test",
        type=str,
    )

    parser.add_argument(
        "-prelim",
        "--prelim_path",
        metavar="prelim_path",
        help="path where the preliminary dataframe is.",
        default="data/qm_calculations/df_prelim_calc_test.pkl",
        type=str,
    )

    parser.add_argument(
        "-result",
        "--result_path",
        metavar="result_path",
        help="path where the resulting dataframe is placed.",
        default="data/qm_calculations/df_results_calc_test.pkl",
        type=str,
    )

    args = parser.parse_args()

    return args


def get_calc_error_timing(
    path_folder,
):
    """Get the timing and error information from the submitit output files."""

    # path_submitit = Path(Path.home() / "pKalculator" / "src" / "pKalculator")
    # submitit_folder = 'submitit_pKalculator_2_alpb_DMSO_sp_cpcm_bordwellCH_3'

    paths_out = [path for path in Path(f"{path_folder}").glob("*.out")]

    df_timing = pd.DataFrame(
        columns=[
            "out_file, job_id",
            "date",
            "start_time",
            "end_time",
            "duration",
            "cpus",
            "mem",
            "error",
        ]
    )
    df_errors = pd.DataFrame(columns=["out_file", "error"])

    errors = 0
    num_output = sum(1 for _ in paths_out)

    lst_erros = []
    lst_job_date_start_end_dur = []

    for out in paths_out:
        with open(out, "r") as fh:
            output = fh.read()
            # if 'ERROR' or 'Submitted job triggered an exception' or 'WARNING! pKalculator failed'in output:
            #     errors += 1
            #     print(out.name)
            #     continue
            if "WARNING! failed" in output:
                for line in output.strip().split("\n"):
                    if "WARNING! Failed" in line:
                        split_start_line = line.split(" ")
                        error_compound = split_start_line[-2]
                        error_smiles = split_start_line[-1]
                        errors += 1
                        # print(error_compound)
                        lst_erros.append((out.name, error_compound, error_smiles))
                continue
            for line in output.strip().split("\n"):
                if "submitit INFO" and "Starting with JobEnvironment" in line:
                    split_start_line = line.split(" ")
                    start_date = split_start_line[2].replace("(", "")
                    start_time = split_start_line[3].replace(")", "")
                    t1 = pd.to_datetime(start_date + " " + start_time)
                    job_id = split_start_line[7].split("=")[-1].replace(",", "")

                if "submitit INFO" and "Job completed successfully" in line:
                    # print(line.split(' '))
                    split_start_line = line.split(" ")
                    end_date = split_start_line[2].replace("(", "")
                    end_time = split_start_line[3].replace(")", "")
                    t2 = pd.to_datetime(end_date + " " + end_time)
                    duration = t2 - t1
                    duration_minutes = duration.total_seconds() / 60
                    # get duration in min:seconds

                    # duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                    # print(duration.total_seconds())

            lst_job_date_start_end_dur.append(
                (
                    out.name,
                    job_id,
                    start_date,
                    end_date,
                    start_time,
                    end_time,
                    duration,
                    duration_minutes,
                )
            )

    print(f"total jobs: {num_output}")
    print(f"total errors: {errors}")

    df_timing = pd.DataFrame(
        lst_job_date_start_end_dur,
        columns=[
            "out_file",
            "job_id",
            "start_date",
            "end_date",
            "start_time",
            "end_time",
            "duration",
            "duration_minutes",
        ],
    )
    df_errors = pd.DataFrame(
        lst_erros, columns=["out_file", "error_compound", "error_smiles"]
    )

    return df_timing, df_errors


def get_error_message(submitit_folder, log_file):
    # path_submitit = Path(Path.home() / "pKalculator" / "src" / "pKalculator")
    path_log_file = Path(f"{submitit_folder}/{log_file}")
    try:
        with path_log_file.open("r") as f:
            return f.read().strip().splitlines()
    except FileNotFoundError:
        return "Log file not found."


def load_pickles(folder):
    # print(folder)
    return pd.concat(
        [
            pd.read_pickle(pkl)[1].assign(file=pkl.name)
            for pkl in folder.glob("*_result.pkl")
            if type(pd.read_pickle(pkl)[1]) != str
        ],
        ignore_index=True,
    )  # .sort_values(by='file')


def compute_relative_energy(row, energy_col):
    return [
        (
            float("inf")
            if val == float("inf")
            or row[energy_col] == float("inf")
            or val == 99999.0
            or row[energy_col] == 99999.0
            else (val - row[energy_col])
        )
        for val in row[f"lst_{energy_col}_deprot"]
    ]


def process_submitted_files_halator_dep(
    path_submitit: str, prelim_path=None
) -> pd.DataFrame:
    if not Path(path_submitit).is_dir():
        raise ValueError("path is not a directory")

    if prelim_path:
        try:
            df_prelim = pd.read_pickle(prelim_path)
        except FileNotFoundError:
            raise ValueError(f"{prelim_path} is not a valid path to a pickle file")

    df_submitted = load_pickles(path_submitit)
    # df_submitted = pd.read_pickle(path_submitit)

    if prelim_path:
        missing_names = set(df_prelim.names.unique()).difference(
            set(df_submitted.names.unique())
        )
        df_submitted = pd.concat(
            [df_submitted, df_prelim.loc[df_prelim.names.isin(missing_names)]]
        )

    df_submitted["failed_xtb"] = df_submitted["e_xtb"].isnull()
    df_submitted["failed_dft"] = df_submitted["e_dft"].isnull()
    df_submitted["e_xtb"] = df_submitted["e_xtb"].fillna(float("inf"))
    df_submitted["e_dft"] = df_submitted["e_dft"].fillna(float("inf"))
    # df_submitted.rename(columns={"lst_atomsite": "atomsite"}, inplace=True)
    # df_submitted.rename(columns={"lst_atomindex": "atom_index"}, inplace=True)
    df_submitted.rename(columns={"atom_index": "atomindex"}, inplace=True)
    df_submitted["atomindex"] = df_submitted["atomindex"].astype(int)
    # df_submitted["smiles_deprot"] = df_submitted["smiles_deprot_map"]

    df_neutral = df_submitted.query("neutral == True").reset_index(drop=True)

    df_neg = (
        df_submitted.query("neutral == False")
        .groupby("ref_name")
        .agg(
            {
                col: lambda x: x.tolist()
                for col in df_submitted.columns
                if col
                not in [
                    "ref_name",
                    "comment",
                ]  # check here
            }
        )
        .rename(
            columns=lambda x: (
                f"lst_{x}_deprot"
                if x != "ref_mol_smiles_map"
                else "lst_ref_mol_smiles_map"
            )
        )
    )

    # lst_smiles_deprot_map_deprot, 	lst_smiles_deprot_deprot

    # Sort the lists within each row based on 'lst_atomsite'
    for i, row in df_neg.iterrows():
        sorted_idx = [
            idx
            for idx, _ in sorted(enumerate(row.lst_atomsite_deprot), key=lambda x: x[1])
        ]
        for col in df_neg.columns:
            if col.startswith("lst_"):
                df_neg.at[i, col] = [row[col][idx] for idx in sorted_idx]

    df_neutral_merged = pd.merge(
        df_neutral, df_neg, on="ref_name", how="left", suffixes=("", "_new")
    )

    df_neutral_merged["lst_e_rel_xtb"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_xtb",)
    )
    df_neutral_merged["lst_e_rel_dft"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_dft",)
    )

    df_neutral_merged["e_rel_min_xtb"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_xtb"]), axis=1
    )

    df_neutral_merged["e_rel_min_dft"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_dft"]), axis=1
    )

    df_neutral_merged[f"atomsite_min"] = df_neutral_merged.apply(
        lambda row: (
            -1
            if row["e_rel_min_dft"] == float("inf")
            else [
                atomsite
                for atomsite, pka in zip(
                    row["lst_atomsite_deprot"], row["lst_e_rel_dft"]
                )
                if pka == row["e_rel_min_dft"]
            ][0]
        ),
        axis=1,
    )

    drop_cols = [
        "smiles_deprot_map",
        "atomsite",
        "atomindex",
        "ref_mol_smiles_map",
        "lst_gfn_method_deprot",
        "lst_solvent_model_deprot",
        "lst_solvent_name_deprot",
        "smiles_neutral",
    ]
    # "mol": "mol_neutral",
    rename_cols = {
        "e_xtb": "e_xtb_neutral",
        "e_dft": "e_dft_neutral",
        "failed_xtb": "failed_xtb_neutral",
        "failed_dft": "failed_dft_neutral",
        "lst_smiles_deprot_map_deprot": "lst_smiles_map_deprot",
    }
    df_neutral_merged.drop(columns=drop_cols, inplace=True)
    df_neutral_merged.rename(columns=rename_cols, inplace=True)

    return df_neutral_merged


def process_submitted_files_halator(
    path_submitit: str, prelim_path=None
) -> pd.DataFrame:
    if not Path(path_submitit).is_dir():
        raise ValueError("path is not a directory")

    if prelim_path:
        try:
            df_prelim = pd.read_pickle(prelim_path)
        except FileNotFoundError:
            raise ValueError(f"{prelim_path} is not a valid path to a pickle file")

    df_submitted = load_pickles(path_submitit)
    # df_submitted = pd.read_pickle(path_submitit)

    if prelim_path:
        missing_names = set(df_prelim.names.unique()).difference(
            set(df_submitted.names.unique())
        )
        df_submitted = pd.concat(
            [df_submitted, df_prelim.loc[df_prelim.names.isin(missing_names)]]
        )

    # df_submitted["failed_xtb"] = df_submitted["e_xtb"].isnull()
    # df_submitted["failed_dft"] = df_submitted["e_dft"].isnull()
    df_submitted["e_xtb"] = df_submitted["e_xtb"].fillna(float("inf"))
    df_submitted["e_dft"] = df_submitted["e_dft"].fillna(float("inf"))
    df_submitted["failed_xtb"] = df_submitted["e_xtb"].apply(
        lambda x: True if x == float("inf") or x == 99999.0 else False
    )
    df_submitted["failed_dft"] = df_submitted["e_dft"].apply(
        lambda x: True if x == float("inf") else False
    )
    # df_submitted.rename(columns={"lst_atomsite": "atomsite"}, inplace=True)
    # df_submitted.rename(columns={"lst_atomindex": "atom_index"}, inplace=True)
    df_submitted.rename(columns={"atom_index": "atomindex"}, inplace=True)
    df_submitted["atomindex"] = df_submitted["atomindex"].astype(int)
    # df_submitted["smiles_deprot"] = df_submitted["smiles_deprot_map"]

    df_neutral = df_submitted.query("neutral == True").reset_index(drop=True)

    df_neg = (
        df_submitted.query("neutral == False")
        .groupby("ref_name")
        .agg(
            {
                col: lambda x: x.tolist()
                for col in df_submitted.columns
                if col
                not in [
                    "ref_name",
                    "comment",
                ]  # check here
            }
        )
        .rename(
            columns=lambda x: (
                f"lst_{x}_deprot"
                if x != "ref_mol_smiles_map"
                else "lst_ref_mol_smiles_map"
            )
        )
    )

    # lst_smiles_deprot_map_deprot, 	lst_smiles_deprot_deprot

    # Sort the lists within each row based on 'lst_atomsite'
    for i, row in df_neg.iterrows():
        sorted_idx = [
            idx
            for idx, _ in sorted(enumerate(row.lst_atomsite_deprot), key=lambda x: x[1])
        ]
        for col in df_neg.columns:
            if col.startswith("lst_"):
                df_neg.at[i, col] = [row[col][idx] for idx in sorted_idx]

    df_neutral_merged = pd.merge(
        df_neutral, df_neg, on="ref_name", how="left", suffixes=("", "_new")
    )

    df_neutral_merged["lst_e_rel_xtb"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_xtb",)
    )
    df_neutral_merged["lst_e_rel_dft"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_dft",)
    )

    df_neutral_merged["e_rel_min_xtb"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_xtb"]), axis=1
    )

    df_neutral_merged["e_rel_min_dft"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_dft"]), axis=1
    )

    df_neutral_merged[f"atomsite_min"] = df_neutral_merged.apply(
        lambda row: (
            -1
            if row["e_rel_min_dft"] == float("inf")
            else [
                atomsite
                for atomsite, pka in zip(
                    row["lst_atomsite_deprot"], row["lst_e_rel_dft"]
                )
                if pka == row["e_rel_min_dft"]
            ][0]
        ),
        axis=1,
    )

    drop_cols = [
        "smiles_deprot_map",
        "atomsite",
        "atomindex",
        "ref_mol_smiles_map",
        "lst_gfn_method_deprot",
        "lst_solvent_model_deprot",
        "lst_solvent_name_deprot",
        "smiles_neutral",
        "mol",
        "lst_HA_exp_deprot",
        "lst_ref_link_deprot",
        "lst_neutral_deprot",
        "lst_mol_deprot",
    ]
    # check if they intersect. If they do, drop them
    cols_to_drop = set(df_neutral_merged.columns) & set(drop_cols)

    # "mol": "mol_neutral",
    rename_cols = {
        "e_xtb": "e_xtb_neutral",
        "e_dft": "e_dft_neutral",
        "failed_xtb": "failed_xtb_neutral",
        "failed_dft": "failed_dft_neutral",
        "lst_smiles_deprot_map_deprot": "lst_smiles_map_deprot",
    }
    df_neutral_merged.drop(columns=cols_to_drop, inplace=True)
    df_neutral_merged.rename(columns=rename_cols, inplace=True)

    return df_neutral_merged


def get_cm5_desc_vector(smi_name, smi, lst_atom_index, n_shells=6):
    # make sure that smi is canonical. Smiles in our workflow should provide a canonical smiles as Chen.MolToSmiles() by default generates the canonical smiles
    generator = Generator()
    des = (
        "GraphChargeShell",
        {"charge_type": "cm5", "n_shells": n_shells, "use_cip_sort": True},
    )
    try:
        cm5_list = generator.calc_CM5_charges(
            smi=smi, name=smi_name, optimize=False, save_output=True
        )
        (
            atom_indices,
            descriptor_vector,
            mapper_vector,
        ) = generator.create_descriptor_vector(lst_atom_index, des[0], **des[1])
    except Exception:
        descriptor_vector = None

    return cm5_list, atom_indices, descriptor_vector, mapper_vector


def calc_pka_lfer(e_rel: float) -> float:
    # pka = 0.5941281 * e_rel - 159.33107321
    pka = 0.59454292 * e_rel - 159.5148093
    return pka


def pka_dmso_to_pka_thf(pka: float, reverse=False) -> float:
    pka_thf = -0.963 + 1.046 * pka
    if reverse:
        pka_dmso = (pka_thf + 0.963) / 1.046
        return pka_dmso

    return pka_thf


def pred_HA(df, coef_xtb, intercept_xtb, coef_dft, intercept_dft):

    if coef_xtb and intercept_xtb is not None:
        df["HA_qmpred_xtb"] = df["lst_e_rel_xtb"].apply(
            lambda x: [coef_xtb * energy + intercept_xtb for energy in x]
        )
        df["HA_min_qmpred_xtb"] = df.apply(
            lambda row: min(row["HA_qmpred_xtb"]), axis=1
        )

        df["HA_qmpred_error_xtb"] = df.apply(
            lambda row: [
                abs(row["HA_exp"] - ha_pred) for ha_pred in row["HA_qmpred_xtb"]
            ],
            axis=1,
        )

        df["HA_min_qmpred_error_xtb"] = df.apply(
            lambda row: abs(row["HA_exp"] - min(row["HA_qmpred_xtb"])), axis=1
        )

    df["HA_qmpred_dft"] = df["lst_e_rel_dft"].apply(
        lambda x: [coef_dft * energy + intercept_dft for energy in x]
    )

    df["HA_min_qmpred_dft"] = df.apply(lambda row: min(row["HA_qmpred_dft"]), axis=1)

    df["HA_qmpred_error_dft"] = df.apply(
        lambda row: [abs(row["HA_exp"] - ha_pred) for ha_pred in row["HA_qmpred_dft"]],
        axis=1,
    )

    df["HA_min_qmpred_error_dft"] = df.apply(
        lambda row: abs(row["HA_exp"] - min(row["HA_qmpred_dft"])), axis=1
    )

    return df


if __name__ == "__main__":
    args = get_args()

    calc_path = args.calc_path
    submit_path = args.submit_path
    prelim_path = args.prelim_path
    result_path = args.result_path

    # path_qm_calculations = Path.cwd() / "data/qm_calculations"
    # path_submitit = path_qm_calculations.joinpath("submit_test")
    # path_calc = path_qm_calculations.joinpath("calc_test")

    path_qm_calculations = Path.cwd() / "data/qm_calculations"
    path_calc = Path.cwd().joinpath(calc_path)
    path_submitit = Path.cwd().joinpath(submit_path)
    prelim_path = Path.cwd().joinpath(prelim_path)
    result_path = Path.cwd().joinpath(result_path)

    print(path_submitit)
    df_results = process_submitted_files(
        path_submitit=path_submitit,
        prelim_path=prelim_path,
    )

    print("calculating cm5 charges and descriptor vectors")
    output = df_results.apply(
        lambda row: get_cm5_desc_vector(
            smi_name=row["names"],
            smi=row["smiles"],
            lst_atom_index=row[
                "lst_atomindex_deprot"
            ],  # lst_atom_index , lst_atom_index_deprot,
            n_shells=6,
        ),
        axis=1,
    )
    (
        df_results["cm5"],
        df_results["atom_indices"],
        df_results["descriptor_vector"],
        df_results["mapper_vector"],
    ) = zip(*output)

    print("saving results to pickle")
    # save df_results
    df_results.to_pickle(result_path)
