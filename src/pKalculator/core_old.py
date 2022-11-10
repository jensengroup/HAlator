# MIT License
#
# Copyright (c) 2022 Nicolai Ree
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import copy
import numpy as np
import pandas as pd
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import molecule_formats as molfmt
import run_xTB as run_xTB
from modify_smiles import change_mol_v6

# CPU and memory usage
num_cpu_single = 8 # number of cpus per job.
mem_gb = 10 # total memory usage per task.




def confsearch_xTB(conf_complex_mols, conf_names, chrg=0, spin=0, method='ff', solvent='', conf_cutoff=10, precalc_path=None):
    
    global num_cpu_single
    
    # Run a constrained xTB optimizations  
    confsearch_args = []
    for i in range(len(conf_names)):
        if precalc_path:
            confsearch_args.append((conf_names[i]+'_gfn'+method.replace(' ', '')+'.xyz', conf_complex_mols[i], chrg, spin, method, solvent, True, precalc_path[i]))
        else:
            confsearch_args.append((conf_names[i]+'_gfn'+method.replace(' ', '')+'.xyz', conf_complex_mols[i], chrg, spin, method, solvent, True, None))

    with ThreadPoolExecutor(max_workers=num_cpu_single) as executor:
        results = executor.map(run_xTB.run_xTB, confsearch_args)

    conf_energies = []
    conf_paths = []
    for result in results:
        conf_energy, path_opt = result
        conf_energies.append(conf_energy)
        conf_paths.append(path_opt)

    # Find the conformers below cutoff #kcal/mol (3 kcal/mol = 12.6 kJ/mol)
    rel_conf_energies = np.array(conf_energies) - np.min(conf_energies) #covert to relative energies
    below_cutoff = (rel_conf_energies <= conf_cutoff).sum() #get number of conf below cutoff

    conf_tuble = list(zip(conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies)) #make a tuble
    conf_tuble = sorted(conf_tuble, key=itemgetter(4))[:below_cutoff] #get only the best conf below cutoff

    conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies = zip(*conf_tuble) #unzip tuble
    conf_names, conf_complex_mols, conf_paths, conf_energies = list(conf_names), list(conf_complex_mols), list(conf_paths), list(conf_energies) #tubles to lists
    mol_files = [os.path.join(conf_path, conf_name+'_gfn'+method.replace(' ', '') + '_opt.sdf') for conf_path, conf_name in zip(conf_paths,conf_names)] #list of paths to optimized structures in .sdf format

    # Find only unique conformers
    conf_names, conf_complex_mols, conf_paths, conf_energies = zip(*molfmt.find_unique_confs(list(zip(conf_names, conf_complex_mols, conf_paths, conf_energies)), mol_files, threshold=0.5)) #find unique conformers
    conf_names, conf_complex_mols, conf_paths, conf_energies = list(conf_names), list(conf_complex_mols), list(conf_paths), list(conf_energies) #tubles to lists

    return conf_names, conf_complex_mols, conf_paths, conf_energies


def calculateEnergy(args):
    """ Embed the post-insertion complex and calculate the ground-state free energy 
    return: energy [kcal/mol]
    """
    
    global num_cpu_single

    rdkit_mol, name, method, solvent_model, solvent_name = args
    method = ' '+str(method)
    solvent = '--'+solvent_model+' '+solvent_name

    # rdkit_mol, name = args
    # method=' 2'  # <--- change the method for accurate calculations ('ff', ' 0', ' 1', ' 2')
    # solvent = '--alpb DMSO' # <--- change the solvent ('--gbsa solvent_name', '--alpb solvent_name', or '')
    chrg = Chem.GetFormalCharge(rdkit_mol) # checking and adding formal charge to "chrg"
    # OBS! Spin is hardcoded to zero!

    # RDkit conf generator
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rot_bond = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdkit_mol)
    n_conformers = min(1 + 3 * rot_bond, 20)
    # p = AllChem.ETKDGv3()
    # p.randomSeed = 90
    # p.useSmallRingTorsions=True
    # p.ETversion=2
    # p.useExpTorsionAnglePrefs=True
    # p.useBasicKnowledge=True
    # AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=n_conformers, params=p)
        
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=n_conformers,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True, ETversion=2, randomSeed=90)

    # Unpack confomers and assign conformer names
    conf_mols = [Chem.Mol(rdkit_mol, False, i) for i in range(rdkit_mol.GetNumConformers())]
    conf_names = [name + f'_conf{str(i+1).zfill(2)}' for i in range(rdkit_mol.GetNumConformers())] #change zfill(2) if more than 99 conformers
    conf_names_copy = copy.deepcopy(conf_names)

    # Run a GFN-FF optimization
    conf_names, conf_mols, conf_paths, conf_energies = confsearch_xTB(conf_mols, conf_names, chrg=chrg, spin=0, method='ff', solvent=solvent, conf_cutoff=10, precalc_path=None)
    
    # Run a GFN?-xTB optimization
    conf_names, conf_mols, conf_paths, conf_energies = confsearch_xTB(conf_mols, conf_names, chrg=chrg, spin=0, method=method, solvent=solvent, conf_cutoff=10, precalc_path=conf_paths)
    
    # Run Orca single point calculations
    final_conf_energies = []
    final_conf_mols = []
    for conf_name, conf_mol, conf_path, conf_energy in zip(conf_names, conf_mols, conf_paths, conf_energies):
        # if conf_energy != 60000.0: # do single point calculations on all unique conformers
        #     conf_energy = run_orca.run_orca('xtbopt.xyz', chrg, os.path.join("/".join(conf_path.split("/")[:-2]), 'full_opt', conf_name+'_full_opt'), ncores=num_cpu_single, mem=(int(mem_gb)/2)*1000)
        final_conf_energies.append(conf_energy)
        final_conf_mols.append(conf_mol)
    
    # Get only the lowest energy conformer
    minE_index = np.argmin(final_conf_energies)
    best_conf_mol = final_conf_mols[minE_index]
    best_conf_energy = final_conf_energies[minE_index] # uncomment when doing single point calculations on all unique conformers
    # best_conf_energy = run_orca.run_orca('xtbopt.xyz', chrg, os.path.join(os.getcwd(), 'calc', conf_names[minE_index]+'_full_opt'), ncores=num_cpu_single, mem=(int(mem_gb)/2)*1000) # comment when doing single point calculations on all unique conformers, otherwise this runs a Orca single point calculation on the lowest xTB energy conformer

    ### START - CLEAN UP ###
    for conf_name in conf_names_copy:

        conf_path = os.path.join(os.getcwd().replace('/src/pKalculator', ''), 'calc', conf_name.split('_')[0], conf_name.split('_')[1])
        
        if os.path.isfile(os.path.join(conf_path, conf_name+'_gfnff.xyz')):
            os.remove(os.path.join(conf_path, conf_name+'_gfnff.xyz'))
        
        if os.path.isfile(os.path.join(conf_path, conf_name+'_gfn'+ method.replace(' ', '') + '.xyz')):
            os.remove(os.path.join(conf_path, conf_name+'_gfn'+ method.replace(' ', '') + '.xyz'))
        
        # Remove GFNFF-xTB folder
        folder_path = os.path.join(conf_path, 'gfnff')
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
            # checking whether the folder is empty or not
            if len(os.listdir(folder_path)) == 0:
                os.rmdir(folder_path)
            else:
                print("Folder is not empty")
    
        # Clean GFN?-xTB folder
        folder_path = os.path.join(conf_path, 'gfn' + method.replace(' ', ''))
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if file_remove.split('.')[-1] in ['sdf', 'xtbout', 'xyz'] and file_remove != 'xtbscreen.xyz':
                    continue
                elif os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
    ### END - CLEAN UP ###

    return best_conf_energy, best_conf_mol


def calc_pKa(E_rel, T=None, pKa_ref=None):
    if T == None:
        T = 298.15 #K
    
    R = 1.987 #kcal/(mol*K) or 8.31441 J/(mol*K) gas constant 

    pKa = pKa_ref + (E_rel/(R*T*np.log(10)))

    return pKa   


def calc_E_rel(input_smiles, name, rxn_type='rm_proton', method=2, solvent_model='alpb', solvent_name='DMSO'):
    lst_E = []
    lst_E_rel = []
    lst_pKa_deprot = []
    input_mol = Chem.MolFromSmiles(input_smiles)

    E_neutral, _ = calculateEnergy((input_mol, name, method, solvent_model, solvent_name))
    E_neutral = round(E_neutral, 2)
    # print(name)
    # print(f'E_neutral = {E_neutral}')
    

    lst_name_deprot, _, lst_smiles_deprot, _, _ = change_mol_v6(name=name, smiles=input_smiles, rxn=rxn_type, reduce_smiles=True, show_mol=False)
    lst_mol_deprot = [Chem.MolFromSmiles(smi) for smi in lst_smiles_deprot]

    for name, mol in list(zip(lst_name_deprot, lst_mol_deprot)):
        E_deprot, _ = calculateEnergy((mol, name, method, solvent_model, solvent_name))
        lst_E.append(round(E_deprot, 2))
        E_rel = round(E_deprot - E_neutral, 2)
        lst_E_rel.append(E_rel)
        pKa_deprot = round(calc_pKa(E_rel=E_rel, pKa_ref=35), 2)
        lst_pKa_deprot.append(pKa_deprot)
        # print(f'd_G = {E_rel:.2f} kcal/mol')
        # print(f'pKa_deprot = {pKa_deprot}')
    
    return E_neutral, lst_name_deprot, lst_E, lst_E_rel, lst_pKa_deprot

def merge_dicts(dict1, dict2):
    return dict1.update(dict2)

def control_calcs(df, xtb_params):
    df = df.copy()

    df['E_neutral'] = pd.Series(dtype='object')
    df['lst_name_deprot'] = pd.Series(dtype='object')
    df['lst_E'] = pd.Series(dtype='object')
    df['lst_E_rel'] = pd.Series(dtype='object')
    df['lst_pKa_deprot'] = pd.Series(dtype='object')
    df['gfn_method'] = pd.Series(dtype='object')
    df['solvent_model'] = pd.Series(dtype='object')
    df['solvent_name'] = pd.Series(dtype='object')

    for idx, row in df.iterrows():
        name = row['name']
        smiles = row['smiles']
        
        params = {
            'input_smiles' : smiles,
            'name' : name, 
            'rxn_type' : 'rm_proton'
            }

        #merges xtb parameters into the params dictionary
        merge_dicts(params, xtb_params)
        print(params)
        
        ### RUN CALCULATIONS ###
        # E_neutral, lst_name_deprot, lst_E, lst_E_rel, lst_pKa_deprot= calc_E_rel(smiles, name, rxn_type='rm_proton', method=2, solvent_model='alpb', solvent_name='DMSO')
        E_neutral, lst_name_deprot, lst_E, lst_E_rel, lst_pKa_deprot= calc_E_rel(**params)
        
        df.at[idx, 'E_neutral'] = E_neutral
        df.at[idx, 'lst_name_deprot'] = lst_name_deprot
        df.at[idx, 'lst_E'] = lst_E
        df.at[idx, 'lst_E_rel'] = lst_E_rel
        df.at[idx, 'lst_pKa_deprot'] = lst_pKa_deprot
        df.at[idx, 'gfn_method'] = 'gfn'+str(params['method'])
        df.at[idx, 'solvent_model'] = params['solvent_model']
        df.at[idx, 'solvent_name'] = params['solvent_name']

    return df


if __name__ == "__main__":

    import pandas as pd
    import submitit
    import time
    from pathlib import Path

    root_path = Path.home()
    print(root_path)
    pKalculator_path = Path(root_path/'pKalculator')

    df = pd.read_csv(Path(pKalculator_path/'test'/'compounds_paper.smiles'), sep=' ') #dataset
    # df = pd.read_csv('../../test/compounds_paper.smiles', sep=' ') #dataset
    df = df.head(2)
    print(df)

    xtb_params = {
            'method' : 2, 
            'solvent_model' : 'alpb',
            'solvent_name' : 'DMSO'
            }

    # #Create a submitit folder for several parameters
    # path = Path(pKalculator_path/'src'/'pKalculator'/'submitit_pKalculator')
    # try:
    #     path.mkdir(mode=0o755, parents=True, exist_ok=True)
    # except FileExistsError:
    #     print("Folder is already there")
    # else:
    #     print("Folder was created")
    
    
    executor = submitit.AutoExecutor(folder="submitit_pKalculator") #Path(path/"submitit_pKalculator_test"
    executor.update_parameters(
        name="pKalculator",
        cpus_per_task=int(num_cpu_single),
        mem_gb=int(mem_gb),
        timeout_min=6000,
        slurm_partition="kemi1", #"kemi1"
        slurm_array_parallelism=50,
    )
    print(executor)

    jobs = []
    with executor.batch():
        chunk_size = 1
        for start in range(0, df.shape[0], chunk_size):
            df_subset = df.iloc[start:start + chunk_size]
            print(df_subset)
            job = executor.submit(control_calcs(df=df, xtb_params=xtb_params), df_subset)
            jobs.append(job)

    
    # start = time.perf_counter()

    # # input_smiles = 'c1c(c2cc(sc2)C)n[nH]c1'
    # # input_mol = Chem.MolFromSmiles(input_smiles)
    # # name = 'test'

    # # best_conf_energy, best_conf_mol = calculateEnergy((input_mol, name))
    # # print(best_conf_energy)
    
    # input_smiles = 'O1C=CC=C1'
    # input_mol = Chem.MolFromSmiles(input_smiles)
    # name = 'furan' 

    # lst_name_deprot, lst_E_rel, lst_pKa_deprot = calc_E_rel(input_smiles, name, rxn_type='rm_proton')
    # print(list(zip(lst_name_deprot, lst_E_rel, lst_pKa_deprot)))
    # for name_deprot, E_rel in list(zip(lst_name_deprot, lst_E_rel)):
    #     print(f'{name_deprot}, pKa = {calc_pKa(E_rel=E_rel, pKa_ref=35)}')

    # finish = time.perf_counter()
    # print(f'Finished in {round(finish-start, 2)} second(s)')
