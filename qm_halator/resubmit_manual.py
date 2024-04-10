from run_orca import run_orca
import pandas as pd
from rdkit import Chem
from pathlib import Path
import submitit
import os
import sys

# best_conf_energy_sp = run_orca(
#             xyz_file="xtbopt.xyz",
#             chrg=chrg,
#             path=conf_paths[minE_index]
#             ncores=num_cpu_single / 2,
#             mem=(int(mem_gb)) * 1000,
#             functional=functional,
#             basis_set=basis,
#             dispersion=dispersion,
#             optfreq=optfreq,
#             solvent_name=solvent_name,
#         )

current_directory = Path.cwd()
# Get the parent directory until you reach 'HAlator'
home_directory = current_directory
while home_directory.name != "HAlator":
    if home_directory == home_directory.parent:  # We've reached the root directory
        raise Exception("HAlator directory not found")
    home_directory = home_directory.parent

# if home_directory.name != "HAlator":
#     raise ValueError("Please run this script from the pKalculator directory")
sys.path.append(str(home_directory / "qm_halator"))
os.chdir(home_directory / "qm_halator")


# load prelim dataframe for compounds that needs to be resubmitted

df = pd.read_pickle(
    "/groups/kemi/borup/HAlator/data/qm_calculations/df_prelim_calc_HA_optfreq_camb3lypd4_exp_val_rerun_20240327.pkl"
)
names_resubmit = [
    "val3#1=4",
    "val4#1=4",
    "val4#1=6",
    "val4#1=15",
    "val4#1=20",
    "val24#1=9",
]
# [
#     'val2#1=5',
#     'val2#1=6',
#     'val2#1=8',
#     'val24#1=3',
#     'val24#1=9',
#     'val3#1=4',
#     'val4#1=1',
#     'val4#1=4',
#     'val4#1=6',
#     'val4#1=15',
#     'val4#1=20'
#     ]


df_merged = df[df["names"].isin(names_resubmit)].copy()
# print(df)

functional = "CAM-B3LYP"
basis_set = "def2-TZVPPD"
dispersion = True
opt = True
freq = True
solvent_name = "DMSO"
num_cpu_single = 20
mem_gb = 60

path_resubmit = Path(
    "/groups/kemi/borup/HAlator/submit_HA_optfreq_camb3lypd4_exp_val_manual"
)


def control_calcs(df):
    new_columns = [
        "mol",
        "e_xtb",
        "e_dft",
        "gfn_method",
        "solvent_model",
        "solvent_name",
    ]

    for col in new_columns:
        df[col] = None

    # Store values in a dictionary
    results = {}

    # ensures that leading or trailing whitespace is removed from column names
    df.columns = df.columns.str.strip()

    for idx, row in df.iterrows():
        name = row["names"]
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        chrg = Chem.GetFormalCharge(mol)
        # print(name)
        # print(chrg, smiles)
        path_calc = Path(
            f"/groups/kemi/borup/HAlator/calc_HA_optfreq_camb3lypd4_exp_val_manual"
        )
        # find xyz file from name]
        # print(name)
        print(idx)
        for path in Path.glob(path_calc, f"{name}_*"):
            print(path)
            print(f"{path.parent}")
            print(f"{path.name}")
            print(f"{path.stem}")
            print("-------------------")

            print("--running Orca--")
            best_conf_energy_sp = run_orca(
                xyz_file=f"{path.name}.xyz",
                chrg=chrg,
                path=path,
                ncores=num_cpu_single / 2,
                mem=(int(mem_gb)) * 1000,
                functional=functional,
                basis_set=basis_set,
                dispersion=dispersion,
                opt=opt,
                freq=freq,
                solvent_name=solvent_name,
                manual=True,
            )

            values = {
                "e_dft": best_conf_energy_sp,
            }

            results[idx] = values

    for idx, values in results.items():
        for col, value in values.items():
            df.at[idx, col] = value

    return df


# # load the xyz file

# print(control_calcs(df=df_merged))

executor = submitit.AutoExecutor(folder=path_resubmit)

executor.update_parameters(
    name="HAlator",  # pKalculator
    cpus_per_task=int(num_cpu_single),
    mem_gb=int(mem_gb),
    timeout_min=60000,  # 500 hours -> 20 days : 60000 --> 1000 hours -> 41 days
    slurm_partition="kemi1",
    slurm_array_parallelism=20,
)
print(executor)

jobs = []
with executor.batch():
    chunk_size = 1
    for start in range(0, df_merged.shape[0], chunk_size):
        df_subset = df_merged.iloc[start : start + chunk_size]
        job = executor.submit(control_calcs, df_subset)
        jobs.append(job)
