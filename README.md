# pKalculator
pKalculator is a fully automated quantum chemistry (QM)-based workflow that computes the C-H pKa values of molecules. The QM workflow uses GNF2-xTB with ORCA on top.
pKalculator also includes an atom-based machine learning model (ML) to predict the C-H pKa values. The ML model (LightGBM regression model) is based on CM5 atomic charges that are computed using semiempirical tight binding (GFN1-xTB). 

For more, see [pKalculator: A pKa predictor for C-H bonds](https://www.google.com)

## Installation
We recommend using `conda` to get the required dependencies

    conda env create -f environment.yml && conda activate pkalculator

Download the latest version of xtb (v. 6.7.0)

    cd dep; wget https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz; tar -xvf ./xtb-6.7.0-linux-x86_64.tar.xz; cd ..


Hereafter, ORCA (v. 5.0.4) is required. Installation instructions can be found at https://sites.google.com/site/orcainputlibrary/setting-up-orca

ORCA requires a specific path for our QM workflow to work. Therefore, follow the comments in "src/qm_pkalculator/run_orca.py" and modify the path accordingly.

## Usage
Both our QM workflow and ML workflow are accessible through the command line in the terminal.

### QM workflow
Below is an example of how to start the QM workflow:

    python src/pkalculator/qm_pkalculator.py -cpus 5 -mem 10 -csv test.smiles -calc calc_test -submitit submitit_test -f CAM-B3LYP -b def2-TZVPPD -s DMSO -d -o -f

The arguments are explained below
-cpus : Number of cpus per job
-mem : Memory in GB per job
-csv : csv path. The csv file must be comma seperated and contain a 'names' column and a 'smiles' column.
-calc : path for saving calculations
-submitit : path for saving results from submitit
-f : The functional to be used
-b : which basis set
-s : solvent for the 
-d : set if D4 dispersion correction
-o : set if optimization is needed
-q : set if frequency computations are required

If needed, SLURM commands can be updated to work at your HPC.

- slurm_partition: SLURM partion.
- timeout_min: The total time that is allowed for each SLURM job before time out.
- slurm_array_parallelism: Maximum number SLURM jobs to run simultaneously (taking one molecule at a time in batch mode).


### ML workflow
Below is an example of how to use the ML workflow:
    
    python src/pkalculator/ml_pkalculator.py -s CC(=O)Cc1ccccc1 -n comp2 -m ml_data/models/full_models/reg_model_all_data_dart.txt

The arguments are explained below:
-s : SMILES string
-n : Name of the compound
-m : Which model to be used. 
-e : Identify the possible site of reaction within (e) pKa units of the lowest pKa value

Hereafter, a list of tuples are returned [(0, 23.14), (3, 18.78), (5, 42.42), (6, 42.9), (7, 43.27)]. The first element in each tuple is the atom index and the second element in each tuple is the ML predicted pKa value for that atom index.

The workflow then produces an .png or .svg image of the molecule with its atom indices for easy comparison. The image of the molecule will also contain a teal circle that highlights the site with the lowest pKa value or within (e) pKa units from the lowest pKa value.



## Citation
