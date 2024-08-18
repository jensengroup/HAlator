# HAlator
HAlator is a fully automated quantum chemistry (QM)-based workflow that computes the C-H hydricities/Hydride affinities of molecules. The QM workflow uses GFN2-xTB with ORCA on top.
HAlator also includes an atom-based machine learning model (ML) to predict the C-H hydricities. The ML model (LightGBM regression model) is based on CM5 atomic charges that are computed using semiempirical tight binding (GFN1-xTB).

For more, see [Predicting C-H activation through hydride affinity and homolytic bond dissociation energies](TBD)

<a href="https://colab.research.google.com/drive/1ohqSGrrVJ6bHJJUg7MdzG7GqM0SnFciq?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Installation
We recommend using `conda` to get the required dependencies

    conda env create -f environment.yml && conda activate HAlator

We recommend downloading the precompiled binaries for the latest version of xTB (v. 6.7.0)

    mkdir dep; cd dep; wget https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz; tar -xvf ./xtb-6.7.0-linux-x86_64.tar.xz; mv xtb-dist xtb-6.7.0; cd ..

If this does not work for your system, xTB can also be installed through `conda`
    conda install -c conda-forge xtb

For more information, please see: https://xtb-docs.readthedocs.io/en/latest/setup.html

Hereafter, ORCA (v. 5.0.4) is required for the QM workflow. Installation instructions can be found at https://www.orcasoftware.de/tutorials_orca/first_steps/install.html, https://sites.google.com/site/orcainputlibrary/setting-up-orca, and https://www.faccts.de/docs/orca/5.0/tutorials/.

ORCA requires a specific path for our QM workflow to work. Therefore, change the paths under the function `run_orca_calculation` in `qm_halator/run_orca.py`.



## Usage
Both our QM workflow and ML workflow are accessible through the command line in the terminal.

### QM workflow
#### QM calculations
Below is an example of how to start the QM workflow:

    python qm_halator/qm_halator.py -f r2scan-3c -o -q

This will start the QM workflow with a the test.csv file located under `data/qm_calculations/test.csv`. For the full example, see below:

    python qm_halator/qm_halator.py -cpus 4 -mem 8 -csv data/test.csv -calc data/qm_calculations/calc_test -submit data/qm_calculations/submit_test -f r2scan-3c -o -q

The arguments for `qm_halator.py` are explained below:
| Arguments    | Description | 
| :------- |:---------|
| `-cpus` | Number of cpus per job. Defaults to 4 cpus |
| `-mem` | Amount of memory per job in GB. Defaults to 8 GB |
| `-p` | Set the SLURM partion to be used at your HPC. Defaults to kemi1 |
| `-csv` | CSV path. The csv file must be comma seperated and contain a 'names' column and a 'smiles' column. Defaults to "data/qm_calculations/test.csv"|
| `-calc` | Path for saving calculations. Defaults to "data/qm_calculations/calc_test" |
| `-submit` | Path for saving results from submitit. Defaults to "data/qm_calculations/submit_test" |
| `-f` | The functional to be used. Defaults to 'r2scan-3c' |
| `-b` | which basis set. Defaults to 'def2-TZVPPD' when invoked |
| `-s` | solvent for the. Defaults to 'DMSO' |
| `-d` | Set if D4 dispersion correction. This is recommended for other functionals than r2SCAN-3C DFT calculations. |
| `-o` | Set if optimization is needed. This is recommended.|
| `-q` | Set if frequency computations are required. This is recommended. |

If needed, SLURM commands can be updated to work at your HPC.

- timeout_min| The total time that is allowed for each SLURM job before time out.
- slurm_array_parallelism| Maximum number SLURM jobs to run simultaneously.

#### Producing the dataframe with results
The QM workflow produces a preliminary dataframe with both the neutral smiles and deprotonated smiles that is needed to run for determining the QM computed C-H hydricities.

The default location is here: `data/qm_calculations/`.

After the QM calculations are completed, please run: 
    
    python qm_pkalculator/etl.py

Now, the resulting dataframe with QM calculations are produced with the default location is here: `data/qm_calculations/`.

The arguments for `etl.py` are explained below:
| Arguments    | Description | 
| :------- |:---------|
| `-calc` | Path for saving calculations. Defaults to "data/qm_calculations/calc_test" |
| `-submit` | Path for saving results from submitit. Defaults to "data/qm_calculations/submit_test" |
| `-prelim` | path where the preliminary dataframe is. Defaults to "data/qm_calculations/df_prelim_calc_test.pkl" |
| `-result` | Path where the resulting dataframe is placed. Defaults to "data/qm_calculations/df_results_calc_test.pkl" |
| `-no_smi2gcs` | Will not calculate the CM5 charges and descriptor vectors for the molecules. By default it the descriptor vectors will be calculated. Use -no_smi2gcs False if the descriptor vectors are not needed." |

### ML workflow
Below is an example of how to use the ML workflow:
    
    python ml_halator/ml_halator.py -s CC(=O)Cc1ccccc1 -n comp2 -m models/reg_model_all.txt

The arguments for the ML workflow are explained below:
| Arguments    | Description | 
| :--- |:---------|
| `-s` | SMILES string. Defaults to 'CC(=O)Cc1ccccc1' |
| `-n` | Name of the compound. Defaults to 'comp2' |
| `-m` | Which model to be used. Defaults to the full regression model |
| `-e` | Identify the possible sites of reaction within {number} kcal/mol of the lowest hydricity. Defaults to 0.0. |

Hereafter, a list of tuples are returned:
    `[(0, 23.14), (3, 18.78), (5, 42.42), (6, 42.9), (7, 43.27)]`

The first element in each tuple is the atom index and the second element in each tuple is the ML predicted hydricity for that atom index.

The workflow then produces an .png or .svg image (default) of the molecule with its atom index for easy comparison. The image of the molecule will also contain a teal circle that highlights the site with the lowest hydricity. If -e is > 0, orange circles highlights the sites that are within {number} kcal/mol from the lowest hydricity. The .png or .svg image is by default saved to `data/ml_predictions/`.

### Data
#### Computed data for CM5 charges 
Both the QM workflow and the ML workflow uses GFN1-xTB to produce CM5 charges. The data from the xTB calculation is saved to  `data/calc_smi2gcs`.

#### Additional data
All additionl data can be found [here](https://sid.erda.dk/sharelink/coKwQQzlzr)

#### Additional data
Here the data is split into three folders: `datasets`, `qm_data` and `ml_data`. The description for each folder is found below:

| Folder    | Description |
| :------- |:---------|
| `datasets` | Includes all datasets. Each `.pkl` contains a pandas DataFrame that can be loaded using the following command `pd.read_pickle(dataets/{dataset name})`. |
| `qm_data/` | Includes all QM calculations, including .xyz files and .log files for either the QM dataset, reaction dataset or the smi2gcs calculations. |
| `ml_data/ALFABET` | Includes the ML predicted BDEs by ALFABET for either the QM dataset or the reaction dataset. |
| `ml_data/models` | Includes trained ML models on either all data or ML models trained on the training set (80 % of the data). |
| `ml_data/validation` | Includes data from the cross-validation for the ML models. The .log files gives an overview of the performance metrics. |

## Citation
TBD