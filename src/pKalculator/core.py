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

# CPU and memory usage
# -- Note, that ORCA is set to use 8 cpu cores and 2 conformers are running in parallel
#    resulting in a total of 16 cpu cores per task. Memory per ORCA calculation is set to (mem_gb/2)*1000 MB.
num_cpu_parallel = 2 # number of parallel jobs.
num_cpu_single = 8 # number of cpus per job.
mem_gb = 40 # total memory usage per task.




def confsearch_xTB(conf_complex_mols, conf_names, chrg=0, spin=0, method='ff', solvent='', conf_cutoff=50, precalc_path=None):
    
    global num_cpu_single
    
    # Run a constrained xTB optimizations  
    confsearch_args = []
    for i in range(len(conf_names)):
        if precalc_path:
            confsearch_args.append((conf_names[i]+'_full_opt.xyz', conf_complex_mols[i], chrg, spin, method, solvent, True, precalc_path[i]))
        else:
            confsearch_args.append((conf_names[i]+'_full_opt.xyz', conf_complex_mols[i], chrg, spin, method, solvent, True, None))

    with ThreadPoolExecutor(max_workers=num_cpu_single) as executor:
        results = executor.map(run_xTB.run_xTB, confsearch_args)

    conf_energies = []
    conf_paths = []
    for result in results:
        conf_energy, path_opt = result
        conf_energies.append(conf_energy)
        conf_paths.append(path_opt)

    # Find the conformers below cutoff #kJ/mol (12.6 kJ/mol = 3 kcal/mol)
    rel_conf_energies = np.array(conf_energies) - np.min(conf_energies) #covert to relative energies
    below_cutoff = (rel_conf_energies <= conf_cutoff).sum() #get number of conf below cutoff

    conf_tuble = list(zip(conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies)) #make a tuble
    conf_tuble = sorted(conf_tuble, key=itemgetter(4))[:below_cutoff] #get only the best conf below cutoff

    conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies = zip(*conf_tuble) #unzip tuble
    conf_names, conf_complex_mols, conf_paths, conf_energies = list(conf_names), list(conf_complex_mols), list(conf_paths), list(conf_energies) #tubles to lists
    mol_files = [os.path.join(item, item.split('/')[-1] + '_opt.sdf') for item in conf_paths] #list of paths to optimized structures in .sdf format

    # Find only unique conformers
    conf_names, conf_complex_mols, conf_paths, conf_energies = zip(*molfmt.find_unique_confs(list(zip(conf_names, conf_complex_mols, conf_paths, conf_energies)), mol_files, threshold=0.5)) #find unique conformers
    conf_names, conf_complex_mols, conf_paths, conf_energies = list(conf_names), list(conf_complex_mols), list(conf_paths), list(conf_energies) #tubles to lists

    return conf_names, conf_complex_mols, conf_paths, conf_energies


def calculateEnergy(args):
    """ Embed the post-insertion complex and calculate the ground-state free energy 
    return: energy [kJ/mol]
    """
    
    global num_cpu_single

    rdkit_mol, name = args
    method=' 2'  # <--- change the method for accurate calculations ('ff', ' 0', ' 1', ' 2')
    solvent = '--alpb Phenol' # <--- change the solvent ('--gbsa solvent_name', '--alpb solvent_name', or '')
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
        
        if os.path.isfile(os.path.join(conf_path, conf_name+'_full_opt.xyz')):
            os.remove(os.path.join(conf_path, conf_name+'_full_opt.xyz'))
        
        # Remove GFNFF-xTB folder
        folder_path = os.path.join(conf_path, 'contrained_gfnff', conf_name + '_contrained_gfnff')
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
            # checking whether the folder is empty or not
            if len(os.listdir(folder_path)) == 0:
                os.rmdir(folder_path)
            else:
                print("Folder is not empty")
    
        # Remove GFN?-xTB folder
        folder_path = os.path.join(conf_path, 'contrained_gfn' + method.replace(' ', ''), conf_name + '_contrained_gfn' + method.replace(' ', ''))
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
            # checking whether the folder is empty or not
            if len(os.listdir(folder_path)) == 0:
                os.rmdir(folder_path)
            else:
                print("Folder is not empty")

        # Clean full opt folder
        folder_path = os.path.join(conf_path, 'full_opt', conf_name + '_full_opt')
        file_remove_list = ['charges', 'coordprot.0', 'lmocent.coord', 'orca_calc_atom46.densities',
                        'orca_calc_atom46.out', 'orca_calc_atom46_property.txt', 'orca_calc_atom53.densities',
                        'orca_calc_atom53.out', 'orca_calc_atom53_property.txt', 'orca_calc.cpcm',
                        'orca_calc.densities', 'orca_calc.gbw', 'wbo', 'xtblmoinfo', 'xtbopt.log',
                        '.xtboptok', 'xtbrestart', 'xtbscreen.xyz']
        if os.path.exists(folder_path):
            for file_remove in file_remove_list:
                if os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
    
    folder_path = os.path.join(conf_path, 'contrained_gfnff')
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    
    folder_path = os.path.join(conf_path, 'contrained_gfn'+method.replace(' ', '')) 
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    ### END - CLEAN UP ###

    return best_conf_energy, best_conf_mol



if __name__ == "__main__":
    
    input_smiles = 'c1c(c2cc(sc2)C)n[nH]c1'
    input_mol = Chem.MolFromSmiles(input_smiles)
    name = 'test'
    
    best_conf_energy, best_conf_mol = calculateEnergy((input_mol, name))
    print(best_conf_energy)