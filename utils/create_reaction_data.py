from rdkit import Chem
import pandas as pd
import numpy as np
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
from modify_smiles import remove_Hs_halator


# ref 38 Role of Sterically Demanding Chiral Dirhodium Catalysts in Site-Selective C–H Functionalization of Activated Primary C–H Bonds

dict_smiles_ref38 = {
    "scheme2": ["CC1=CC=C(CC)C=C1", "CC1=CC=C(C(C)C)C=C1"],
    "table2": [
        "CC1=CC=C(CCC(C)C)C=C1",
        "CC1=CC=C(CC(C)C)C=C1",
        "CC1=CC=C(CCCC)C=C1",
        "CC1=CC=C(CCC)C=C1",
        "CC1=CC=C(OCC(C)C)C=C1",
        "CC1=CC=C(OC)C=C1",
        "CC1=CC=C(OC(CCC2CCCC2)=O)C=C1",
        "CC1=CC=C(C#CCCC(C)C)C=C1",
        "CC1=CC=C(C(OCCCC(C)C)=O)C=C1",
    ],
    "scheme3": ["CC(C)/C=C/C", "CCC/C=C/C"],
    "scheme4": ["C/C=C/C1=CC=C(OC)C=C1"],
    "scheme5": ["CCCCOC"],
    "scheme6": ["CC1=CC[C@]23C(CC[C@H]3C)C(C)(C)C1C2"],
    "scheme7": ["CC1=CC2=CCC(C(CCC3[C@H](C)CCCCC(C)C)[C@]3(C)CC4)C4[C@@]2(C)CC1"],
}

# ref 43 - Late-stage C–H functionalization of complex alkaloids and drug molecules via intermolecular rhodium-carbenoid insertion
dict_smiles_ref43 = {
    "brucine": "O=C(C[C@]1([H])OCC=C2C3)N4C5=CC(OC)=C(OC)C=C5[C@@]67[C@]4([H])[C@@]1([H])[C@H]2C[C@@H]6N3CC7",
    "securinine": "O=C1O[C@@]([C@]2([H])N3CCCC2)(C[C@H]3C=C4)C4=C1",
    "apovincamine1": "[H][C@@]12C3=C4C5=C(N3C(C(OC)=O)=C[C@]1(CC)CCCN2CC4)C=CC=C5",
    "apovincamine2": "[H][C@@]12C3=C4C5=C(N3C(C(OC)=O)=C[C@]1(CC)CCCN2CC4)C=CC(I)=C5",
    "detromethorphan": "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1CC4=C3C=C(OC)C=C4",
    "sercloremine": "CN1CCC(C2=CC3=C(C=CC(Cl)=C3)O2)CC1",
    "thebaine": "COC1=CC=C2C[C@@H](N(CC3)C)C4=CC=C(OC)C5[C@]43C2=C1O5",
    "noscapine": "O=C1O[C@H]([C@@H]2N(CCC3=CC4=C(OCO4)C(OC)=C32)C)C5=C1C(OC)=C(OC)C=C5",
    "bicuculline": "O=C1O[C@@H]([C@@H]2C3=CC4=C(C=C3CCN2C)OCO4)C5=C1C6=C(C=C5)OCO6",
}

# ref 44 - Guiding principles for site selective and stereoselective intermolecular C–H functionalization by donor/acceptor rhodium carbenes
chemdraw_smiles_ref44 = """C1=CCC=CC1
C1CCCO1
C1CCCCC1
CCC(C)C
C=CC1=CC=CC=C1
CC(OC(N1CCCC1)=O)(C)C
C1CCCC1
CC(C(C)C)C
CN1[Si](C)(C)CC[Si]1(C)C
CN(CC1=CC=CC=C1)C(OC(C)(C)C)=O
C/C=C/CO[Si](C)(C)C(C)(C)C
CC/C=C/CC
C1(CC2C3)CC3CC(C2)C1
CC1=CCCCC1
COCCOC
C[Si](OCC1C=CC2=C1C=CC=C2)(C)C(C)(C)C
BrC1=CC2=C(C=CC2CO[Si](C)(C)C(C)(C)C)C=C1
CC(C)C1=CC=CC=C1
CC(C)C1=CC=C(C)C=C1
C/C=C/C1=CC=C(OC(C)=O)C=C1
C/C=C/C1=CC=CC=C1
C/C=C/C1=CC(OC)=C(OC)C(OC)=C1
C/C=C/C=C/CO[Si](C)(C)C(C)(C)C
C/C=C/C=C/COC(C)=O
C[Si](OC/C=C/COC(C)=O)(C)C(C)(C)C
C[Si](OCCCC1=CC=C(OC)C=C1)(C)C
O=C(OCCCC1=CC=C(OC)C=C1)C
CC1=CC=C(O[Si](C)(C)C(C)(C)C)C(OC)=C1
CC(OC(N1CC=CCC1)=O)(C)C
C1=CCCCC1
C[Si](C1=CCCCC1)(C)C
CC([Si](C1=CCCCC1)(C2=CC=CC=C2)C3=CC=CC=C3)(C)C
[H][C@]1([C@@H](C(OC)=O)C2=CC=CC=C2)CCCN1C(OC(C)(C)C)=O
CC(OC(N1CCCC1C(OC)=O)=O)(C)C
CC(OC(N1CCCC1CO[Si](C2=CC=CC=C2)(C(C)(C)C)C3=CC=CC=C3)=O)(C)C""".split(
    "\n"
)

# ref47 - Selective Intermolecular Amination of C-H Bonds at Tertiary Carbon
chemdraw_smiles_ref47 = """CC(C)CCCOCC1=CC=CC=C1
CC(C)CCC(C)(C(OCC)=O)C(OCC)=O
O=C(C1)NC(CC1C[C@@H](OC(C)=O)[C@]2([H])C[C@@H](C)C[C@H](C)C2=O)=O
C[C@@H]1CC[C@@H](C(C)C)[C@H](OC(C(C)(C)C)=O)C1
COC([C@@H](NC(OC(C)(C)C)=O)CC(C)C)=O
BrC1=CC=C(S(NC(CCCC(C)C)C)(=O)=O)C=C1
C[C@@]1(C2)CC[C@@H]2C(C)(C)[C@@H]1OCC3=CC=CC=C3
C[C@]1(C2)C[C@](C[C@@H]2C3)(C)C[C@@H]3C1
CC[C@H]1[C@H](C2=CC=CC=C2)C1""".split(
    "\n"
)

# ref48 - Analyzing Site Selectivity in Rh2(esp)2‑Catalyzed Intermolecular C−H Amination Reactions
chemdraw_smiles_ref48 = """CC(C)CCC1=CC=CC=C1
CC(C)CCC1=CC=C(OC)C=C1
CC(C)CCC1=CC=C(C(F)(F)F)C=C1
CC(C)CCC1=CC=C(C(C)(C)C)C=C1
CC(C)CCC1=CC=C(Br)C=C1
CC(C)CCC1=CC=CC(Cl)=C1
CC(C)CCC1=CC=C(C2=CC=CC=C2)C=C1
CC(C)CCCCC1=CC=CC=C1
CC(C)C1=CC=C([C@](CCC[C@]2(CNC(CC(C)CCCC(C)C)=O)[H])(C)[C@@]2([H])CC3)C3=C1""".split(
    "\n"
)


# ref51 - Tropylium Ion Mediated r-Cyanation of Amines

chemdraw_smiles_ref51 = """CC(C)CN(CC(C)C)CC1=CC=CC=C1
CC(C)CN(CC(C)C)CC1=CC=C([N+]([O-])=O)C=C1
CC(C)CN(CC(C)C)CC1=CC=C(OC)C=C1
CN(C)CC1=C(C)C=C(C)C=C1C
CC1(CN(C)C)CCCCC1
C[C@@]12[C@@](CN(C)C2)(C)CCCC1
C=CCN(CC(C)C)CC(C)C
CC(C)CN(CC(C)C)CC(OC)=O
[H][C@]12N(C[C@@H]3C[C@H]2CN4[C@]3([H])CCCC4)CCCC1
C1(CN(CC2=CC=CC=C2)CC3=CC=CC=C3)=CC=CC=C1
CN1CCC2=CC=CC=C2C1
CN(C)CC1=CC=CC=C1""".split(
    "\n"
)

# ref52 - Photoredox activation and anion binding catalysis in the dual catalytic enantioselective synthesis of β-amino esters
chemdraw_smiles_ref52 = """C12=CC=CC=C1CCN(C3=CC=CC=C3)C2
COC(C=C1)=CC=C1N2CC3=CC=CC=C3CC2
COC(C=C1OC)=CC=C1N2CC3=CC=CC=C3CC2
BrC(C=C1)=CC=C1N2CC3=CC=CC=C3CC2
ClC(C=C1)=CC=C1N2CC3=CC(OC)=C(OC)C=C3CC2
ClC1=C2C(CN(C3=CC=CC=C3)CC2)=CC=C1
ClC(C=C1)=CC=C1N2CC3=CC=CC=C3CC2
COC1=C(OC)C=C(CN(C2=CC=CC=C2OC)CC3)C3=C1
COC1=C(OC)C=C(CN(C2=CC=CC=C2)CC3)C3=C1
BrC1=CC=CC=C1N2CC3=CC=CC=C3CC2
ClC1=C2C(CN(C3=CC=C(Cl)C=C3)CC2)=CC=C1
COC1=CC=CC=C1N2CC3=CC=CC=C3CC2""".split(
    "\n"
)

chemdraw_smiles_ref53 = [
    "CN(C(C1N2[C@]3([H])C(C4=CC=CC=C4N3S(=O)(C5=CC=CC=C5)=O)([C@@]67C8=CC=CC=C8N(S(=O)(C9=CC=CC=C9)=O)[C@]6([H])N(C([C@H](C)N(C)C%10=O)=O)[C@H]%10C7)C1)=O)[C@@H](C)C2=O"
]


# ref54 - Late Stage Benzylic C−H Fluorination with [18 F]Fluoride for PET Imaging
chemdraw_smiles_ref54 = """CCC1=CC=C(Br)C=C1
CCC1=CC2=CC=CC=C2C=C1
C1(CCC2=CC=CC=C2)=CC=CC=C1
C#CCCCC1=CC=CC=C1
CCC1=CC=CC2=CC=CC=C21
CCC1=CC=C(Cl)C=C1
BrC1=C(CCCCCC)C=CS1
O=C1C2=CC=CC=C2CCC3=C1C=CC=C3
O=C(CCC1)C2=C1SC=C2
BrC1=CC=CC(CC)=C1
O=C(CCCC1=CC=CC=C1)OC
N#CCCCC1=CC=CC=C1
O=C1CCCC2=CC(OC)=C(OC)C=C21
CCC1=CC=C(I)C=C1
O=C1C2=C(C=CC=C2)C(N1CCCC3=CC=CC=C3)=O
CCC1=C(Br)C=CC=C1
BrCCCCC1=CC=CC=C1
O=C(C(F)(F)F)N1C2=CC=CC=C2CCC1
CCC(C=C1)=CC=C1C2=CC=CC=C2
CCC1=CC=C(OC(C)=O)C=C1
CC(C(OC)=O)C1=CC=C(CC(C)C)C=C1
O=C(C(F)(F)F)N(CC#C)C1CCC2=CC=CC=C21
CC(CCC1=CC2=CC=C(OC)C=C2C=C1)=O
CC1(C)CCC2=C1C=C(C(C)(C)C)C=C2C(C)=O
CC(C=C1)=CC=C1C2=CC(C(F)(F)F)=NN2C3=CC=C(S(=O)(C)=O)C=C3
O=C(OCC)[C@@H](N(C(C(F)(F)F)=O)CC(N1CCC[C@H]1C(OCC)=O)=O)CCC2=CC=CC=C2
CCCCCCCC1=CC=C(CCC(COC(C)=O)(NC(C)=O)COC(C)=O)C=C1
O=C(OC1=C(OC(C)=O)C=CC(CCNC(OC(C)(C)C)=O)=C1)C
C[C@H](C1=C(C=CC=C2)C2=CC=C1)N(C(OC(C)(C)C)=O)CCCC3=CC(C(F)(F)F)=CC=C3""".split(
    "\n"
)

# ref56 - Direct a-Arylation of Ethers through the Combination of Photoredox-Mediated C-H Functionalization and the Minisci Reaction
chemdraw_smiles_ref56 = """BrC1=CN=CC2=CC=CC=C21
BrC1=CC=C(C=NC=C2)C2=C1
ClC1=NC2=CC=CC=C2N=C1
ClC1=NC2=CC=CC=C2C=C1
CC1=NC2=CC=CC=C2C=C1
C12=CC=CC=C1C=CC(C3=CC=CC=C3)=N2
CC1=CC=NC2=CC=CC=C21
ClC1=CC=NC(C)=C1
N#CC1=CC=NC=C1
O=C(C1=CC=CN=C1)OC
O=C(C1=NC=CC=C1)OCC
C1(C2=CC=CC=C2)=CC=CC(C3=CC=CC=C3)=N1
CC1=NC=NC(C)=C1
CC1=NC(N)=NC=C1
CC1=CC=C(Cl)N=N1
C1CCCOC1
C1CCOC1
CC1CCCO1
O=C(OCC1CCCO1)C
O=C1CCOC1
O=C1CCOCC1
C1CCO1
C1OCCOC1
O1CCOC1
CCOCC
CCCCOCCCC
COCCOC""".split(
    "\n"
)

# ref59 - Metal-catalysed azidation of tertiary C–H bonds suitable for late-stage functionalization
chemdraw_smiles_ref59 = """[H][C@@]12[C@@](CCCC2)([H])CCCC1
[H][C@@]12[C@](CCCC2)([H])CCCC1
CC1[C@@H]2C(C)(C)[C@@H](C2)CC1
CC1CCCCC1
CC(C)CC(C)(C)C
CC(C)CCC(C)(C)C
C12CC(CC3C2)CC(C3)C1
CC(C)(C)C1CCCCC1
CCC1=CC=CC=C1
CC(C)C1=CC=CC=C1
C1CCCCC1
CC(C)CCCC(C)COC(C)=O
CC(C)CCCC(C)CCOC(C)=O
CC(C)CCCC(C)CCBr
CC(C)CCCC(C)CCC#N
CC(C)CCCC(C)CC(OC)=O
CC(C)CCCC(C)CC(O)=O
CC(C)CCCC(C)CC(N)=O
O=C1CCC(C(C)C)CC1
CC1=C(CC)C=CC=C1
CC1=CC=C(C(C)C)C=C1
O=C1[C@]2(C)[C@@H](O2)C[C@H](C(C)C)C1
O=C1[C@]2(C)[C@@H](N2C(OCC)=O)C[C@H](C(C)C)C1
O=C1[C@]2(C)[C@@H](C2)C[C@H](C(C)C)C1
C[C@@]1([H])C[C@@H](OC(C)=O)[C@@](C(C)C)([H])CC1
C[C@]12[C@@]([C@](C(OC)=O)(C)CCC2)([H])CCC3=C1C=C(O[Si](C)(C)C(C)(C)C)C=C3
[H][C@]12[C@@]([C@@](CCC3=O)([H])[C@]3(C)CC2)([H])CCC4=C1C=CC(O[Si](C)(C)C(C)(C)C)=C4
O=C1CC[C@]2(C3)[C@@]([C@@](C(O)=O)([H])[C@]4(C[C@]5([H])C)[C@@]2([H])CC[C@]5(O)C4)([H])[C@]1(C)C3=O""".split(
    "\n"
)


# ref123 - Complementation of Biotransformations with Chemical C−HOxidation: Copper-Catalyzed Oxidation of Tertiary Amines in Complex Pharmaceuticals
chemdraw_smiles_ref123 = [
    "OC(C)(C)C(C=C1)=CN=C1N(CC2)CCN2C3=C(C=CC=C4)C4=C(N=N3)CC5=CC=CC=C5"
]

# ref140 - Improving Physical Properties via CH Oxidation: Chemical and Enzymatic Approaches
chemdraw_smiles_ref140 = [
    "C[C@@]12[C@](CC[C@]3(C)[C@]2([H])CC[C@@]4([H])[C@@]3(C)CC[C@]5(CO)[C@]4([H])[C@H](C(C)=C)CC5)([H])C(C)(C)[C@@H](O)CC1"
]

# ref144 - Oxidations by Methyl(trifluoromethyl)dioxirane. 2.1 Oxyfunctionalization of Saturated Hydrocarbons1
chemdraw_smiles_ref144 = """CC1CCCCC1
C[C@@H]1[C@@H](C)CCCC1
[C@H]12CCCC[C@@H]1CCCC2
[C@H]12CCCC[C@H]1CCCC2
[C@@H]1(C2)CC[C@H]2CC1
C1(CC2)CCC2CC1
C12CC(CC3C2)CC(C3)C1
CC(C)C1=CC=CC=C1""".split(
    "\n"
)

# ref Synthetic applications of hydride abstraction reactions by organic oxidants
chemdraw_smiles_ref_synthetic_applications = """OC(C1=CC=CC=C1C(C2=CC=CC=C2)C3=CC=CC=C3)=O
OC(C1=CC=CC=C1CC2=CC=CC=C2)=O
C[C@@H](COCC1=CC=C(OC)C=C1)[C@]([C@@H](C)CC2)([H])O[C@@]2([H])[C@@H](CC)OCC3=CC=CC=C3
CC[C@@H](OCC1=CC=C(OC)C(OC)=C1)[C@H](COCC2=CC=C(OC)C=C2)CCOCC3=CC=CC=C3
C=CCOCC1[C@@H]2[C@@H](OC(C)(C)O2)C(COC=C)O1
C/C(C)=C/COCCCCCOCC=C
COC/C=C/C1=CC=CC=C1
COCC1=CC(OCO2)=C2C=C1
C12=CC=CC=C1CCOC2
C12=CC=CC=C1C=COC2
C12=CC=CC=C1C=CCO2
COC(C=C1)=CC=C1C2C3=CC=CC=C3CCO2
C12=CC=CC=C1CCOC2C3=CC=CC=C3
C12=CC=CC=C1CCN(C3=CC=CC=C3)C2
FC1=CC(C2C(C(OC)=O)=CC(C(OC)=O)=CN2C3=C(O)C=CC=C3)=CC=C1
C1(CC2=CC=CC=C2)=CC=CC=C1
C1(/C=C/CC2=CC=CC=C2)=CC=CC=C1
C[C@@](CC(OC(C)=O)=C)([H])O[C@H](C)C1=CCCCC1
C/C(C)=C\C[C@@H](O)[C@]1(C)C2[C@H](OC)[C@H](O)CC[C@]2(O)CN1C3=CC=C(OC)C=C3
COC1=CC=C(CC2=CC=C(OC)C=C2)C=C1
O=C1C(C)=C[C@](O)([H])[C@](C2(CO2)C[C@@]3([H])O[C@@](/C=C(C)/CC/C=C(C)/C)([H])CC(C)=C3)([H])C1
COC1=C(C(OC)=C(CO[C@H](CCOC2=CC=CC=C2)C3)C3=C4OC)C4=CC=C1
O=C1O[C@@H](CCC)C[C@@H](OC)C/C(C)=C\CO[C@@](CC(OC(C)=O)=C)([H])C1
CO[C@@H]1[C@H](N=[N+]=[N-])[C@@H](OCC2=CC=C(OC)C=C2)[C@H]3[C@@H](CO[C@@H](C4=CC=CC=C4)O3)O1
CC(OC1)(C)O[C@H]1C2[C@H](OCC3=CC=C(OC)C=C3)[C@@H](OC(C)(C)O4)[C@@H]4O2
OC1=CC=C(C(C#CC2=CC=CC=C2)C(F)(F)F)C=C1
BrC(C=C1)=CC=C1C2=CCN(C(OC)=O)CC2
O=C(OCC)CNC1=CC=C(OC)C=C1""".split(
    "\n"
)

# ref18 to Recent Advances in Catalysis Using Organoborane-Mediated Hydride Abstraction -  Direct Conversion of N‑Alkylamines to N‑Propargylamines through C−H Activation Promoted by Lewis Acid/Organocopper Catalysis: Application to Late-Stage
chemdraw_smiles_ref_alpha_funct = """N1(C2=CC=CC=C2)CCCC1
CC1(C)CN(C2=CC=CC=C2)CC1
N1(C2=CC=CC=C2)CCCCCC1
CN(C1=CC=CC=C1)C
CN(C1=CC=CC=C1)CC
CN(C1=CC=CC=C1)CC2=CC=CC=C2
CN(C(CO[Si](C)(C)C(C)(C)C)(C)C)CC1=CC=CC=C1
CN(C(C1=CC=CC=C1)CO[Si](C)(C)C(C)(C)C)CC2=CC=CC=C2
CN(C(C1=CC=CC=C1)CO[Si](C)(C)C(C)(C)C)C(C2=CC=CC=C2)C3=CC=CC=C3
CN(CCC(C1=CC=CC=C1)OC2=CC=C(C(F)(F)F)C=C2)C(C3=CC=CC=C3)C4=CC=CC=C4
CN(CC1=CC=CC2=CC=CC=C12)CC3=CC=C(C(C)(C)C)C=C3
CCC(N(C)C)(C1=CC=CC=C1)COC(C2=CC(OC)=C(C(OC)=C2)OC)=O
CN(C(C1=CC=CC=C1)C2=CC=CC=C2)CC[C@H](C3=CC=CC=C3)OC4=CC=CC=C4C
CN(C(C1=CC=CC=C1)C2=CC=CC=C2)CC/C=C3C4=CC=CC=C4CCC5=CC=CC=C35
CN(C(C1=CC=CC=C1)C2=CC=CC=C2)CC[C@@H](C3=CC=CS3)OC4=CC=CC5=CC=CC=C45
CN(C(C1=CC=CC=C1)C2=CC=CC=C2)[C@H]3CC[C@H](C4=CC=CC=C43)C5=CC(Cl)=C(Cl)C=C5""".split(
    "\n"
)

reaction_sites_ref38 = {
    "carbene0": {
        "atomidx_reaction_electronic": 1,
        "atomidx_reaction_steric": 6,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene1": {
        "atomidx_reaction_electronic": 5,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 8,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 7,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene7": {
        "atomidx_reaction_electronic": 0,
        "atomidx_reaction_steric": 6,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene11": {
        "atomidx_reaction_electronic": 3,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene12": {
        "atomidx_reaction_electronic": 3,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene13": {
        "atomidx_reaction_electronic": 2,
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene14": {
        "atomidx_reaction_electronic": 3,
        "atomidx_reaction_steric": 5,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene15": {
        "atomidx_reaction_electronic": (3, 6),
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene16": {
        "atomidx_reaction_electronic": (5, 28),
        "atomidx_reaction_steric": 0,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref43 = {
    "brucine": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": (12, 28),
    },
    "securinine": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "apovincamine1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 18,
        "atomidx_reaction_minor": None,
    },
    "apovincamine2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 18,
        "atomidx_reaction_minor": None,
    },
    "detromethorphan": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 17,
        "atomidx_reaction_minor": None,
    },
    "sercloremine": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "thebaine": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 20,
        "atomidx_reaction_minor": None,
    },
    "noscapine": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 29,
        "atomidx_reaction_minor": None,
    },
    "bicuculline": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
}


reaction_sites_ref44 = {
    "carbene17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "carbene18": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "carbene19": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene20": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "carbene21": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": None,
        "atomidx_reaction_minor": None,
    },
    "carbene22": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "carbene23": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene24": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "carbene25": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene26": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene27": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "carbene28": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "carbene29": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "carbene30": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "carbene31": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene32": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "carbene33": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "carbene34": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "carbene35": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "carbene36": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene37": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene38": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene39": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "carbene40": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "carbene41": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "carbene42": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "carbene43": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "carbene44": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "carbene45": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "carbene46": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "carbene47": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "carbene48": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "carbene49": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "carbene50": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "carbene51": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref47 = {
    "amination0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "amination1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
    "amination2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 16,
        "atomidx_reaction_minor": None,
    },
    "amination3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "amination4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "amination5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "amination6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "amination7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "amination8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref48 = {
    "amination9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": 8,
    },
    "amination11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": 1,
    },
    "amination16": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": 1,
    },
    "amination17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 28,
        "atomidx_reaction_minor": 1,
    },
}

reaction_sites_ref51 = {
    "cyanation0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "cyanation1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "cyanation2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "cyanation3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "cyanation4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "cyanation5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "cyanation6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "cyanation7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "cyanation8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "cyanation9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "cyanation10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "cyanation11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref52 = {
    "ox_alpha0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 13,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 15,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 15,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 16,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 16,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 15,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 17,
        "atomidx_reaction_minor": None,
    },
    "ox_alpha12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 13,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref53 = {
    "ox1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": (1, 5, 10, 15),
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref54 = {
    "fluorination0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "fluorination3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "fluorination4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "fluorination7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "fluorination8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "fluorination9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "fluorination11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "fluorination12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "fluorination13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 13,
        "atomidx_reaction_minor": None,
    },
    "fluorination15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination16": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "fluorination17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "fluorination18": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination19": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "fluorination20": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "fluorination21": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 12,
        "atomidx_reaction_minor": None,
    },
    "fluorination22": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "fluorination23": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 13,
        "atomidx_reaction_minor": None,
    },
    "fluorination24": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "fluorination25": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 22,
        "atomidx_reaction_minor": None,
    },
    "fluorination26": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 11,
        "atomidx_reaction_minor": None,
    },
    "fluorination27": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "fluorination28": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 15,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref56 = {
    "minisci0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "minisci1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "minisci2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "minisci3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "minisci6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "minisci8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "minisci9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "minisci10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "minisci11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "minisci12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "minisci13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "minisci16": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "minisci17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "minisci18": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "minisci19": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci20": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci21": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "minisci22": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "minisci23": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci24": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "minisci25": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "minisci26": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref59 = {
    "azidation0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "azidation1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "azidation2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "azidation8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "azidation10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "azidation11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": 5,
    },
    "azidation12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 11,
        "atomidx_reaction_minor": 6,
    },
    "azidation13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 6,
    },
    "azidation14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 6,
    },
    "azidation15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": 5,
    },
    "azidation16": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 6,
    },
    "azidation17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 6,
    },
    "azidation18": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 3,
    },
    "azidation19": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 8,
    },
    "azidation20": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": 0,
    },
    "azidation21": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 10,
    },
    "azidation22": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": 6,
    },
    "azidation23": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": 10,
    },
    "azidation24": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 11,
        "atomidx_reaction_minor": (6, 10),
    },
    "azidation25": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 25,
        "atomidx_reaction_minor": 28,
    },
    "azidation26": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": (16, 17, 26),
    },
    "azidation27": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref123 = {
    "ox2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": (9, 16),
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref140 = {
    "ox3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_ref144 = {
    "ox4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "ox5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "ox6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "ox7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "ox8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "ox9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "ox10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
    "ox11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 1,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_synthetic_applications = {
    "appl0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
    "appl1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
    "appl2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 21,
        "atomidx_reaction_minor": None,
    },
    "appl3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "appl4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "appl5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 10,
        "atomidx_reaction_minor": None,
    },
    "appl6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "appl7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 2,
        "atomidx_reaction_minor": None,
    },
    "appl8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
    "appl9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "appl10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 9,
        "atomidx_reaction_minor": None,
    },
    "appl11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "appl12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "appl13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 13,
        "atomidx_reaction_minor": None,
    },
    "appl14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "appl15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "appl16": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 8,
        "atomidx_reaction_minor": None,
    },
    "appl17": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "appl18": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 7,
        "atomidx_reaction_minor": None,
    },
    "appl19": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "appl20": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 14,
        "atomidx_reaction_minor": None,
    },
    "appl21": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 16,
        "atomidx_reaction_minor": None,
    },
    "appl22": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 20,
        "atomidx_reaction_minor": None,
    },
    "appl23": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "appl24": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 6,
        "atomidx_reaction_minor": None,
    },
    "appl25": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "appl26": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "appl27": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
}

reaction_sites_alpha_funct = {
    "afunct0": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "afunct1": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 4,
        "atomidx_reaction_minor": None,
    },
    "afunct2": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 5,
        "atomidx_reaction_minor": None,
    },
    "afunct3": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct4": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 3,
        "atomidx_reaction_minor": None,
    },
    "afunct5": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct6": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct7": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct8": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct9": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct10": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct11": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 26,
        "atomidx_reaction_minor": None,
    },
    "afunct12": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 12,
        "atomidx_reaction_minor": None,
    },
    "afunct13": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct14": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
    "afunct15": {
        "atomidx_reaction_electronic": None,
        "atomidx_reaction_steric": None,
        "atomidx_reaction_major": 0,
        "atomidx_reaction_minor": None,
    },
}


lst_df_ref38 = []
lst_smi_ref38 = [smi for val in dict_smiles_ref38.values() for smi in val]
lenght_smi = len(lst_smi_ref38)
for idx, smi in enumerate(lst_smi_ref38):
    name = f"carbene{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref38.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [38],
                "ref_link": ["https://doi.org/10.1021/ja504797x"],
                "reaction_type": ["carbene insertion"],
            }
        )
    )
df_ref38 = pd.concat(lst_df_ref38, ignore_index=True)

lst_df_ref43 = []
for name, smi in dict_smiles_ref43.items():
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref43.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [43],
                "ref_link": ["https://doi.org/10.1038/ncomms6943"],
                "reaction_type": ["carbene insertion"],
            }
        )
    )
df_ref43 = pd.concat(lst_df_ref43, ignore_index=True)

lst_df_ref44 = []
for idx, smi in enumerate(chemdraw_smiles_ref44):
    # print(smi)
    name = f"carbene{idx+17}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref44.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [44],
                "ref_link": ["https://doi.org/10.1039/C0CS00217H"],
                "reaction_type": ["carbene insertion"],
            }
        )
    )
df_ref44 = pd.concat(lst_df_ref44, ignore_index=True)

lst_df_ref47 = []
for idx, smi in enumerate(chemdraw_smiles_ref47):
    # print(smi)
    name = f"amination{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref47.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [47],
                "ref_link": ["https://doi.org/10.1002/anie.201304238"],
                "reaction_type": ["H_abstraction_C-N"],
            }
        )
    )
df_ref47 = pd.concat(lst_df_ref47, ignore_index=True)

lst_df_ref48 = []
for idx, smi in enumerate(chemdraw_smiles_ref48):
    # print(smi)
    name = f"amination{idx+9}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref48.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [48],
                "ref_link": ["https://doi.org/10.1021/ja5015508"],
                "reaction_type": ["H_abstraction_C-N"],
            }
        )
    )
df_ref48 = pd.concat(lst_df_ref48, ignore_index=True)

lst_df_ref51 = []
for idx, smi in enumerate(chemdraw_smiles_ref51):
    # print(smi)
    name = f"cyanation{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref51.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [51],
                "ref_link": ["https://doi.org/10.1021/ja109617y"],
                "reaction_type": ["oxidation_alpha_heteroatom"],
            }
        )
    )
df_ref51 = pd.concat(lst_df_ref51, ignore_index=True)

lst_df_ref52 = []
for idx, smi in enumerate(chemdraw_smiles_ref52):
    # print(smi)
    name = f"ox_alpha{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref52.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [52],
                "ref_link": ["https://doi.org/10.1039/C3SC52265B"],
                "reaction_type": ["oxidation_alpha_heteroatom"],
            }
        )
    )
df_ref52 = pd.concat(lst_df_ref52, ignore_index=True)

lst_df_ref53 = []
for idx, smi in enumerate(chemdraw_smiles_ref53):
    # print(smi)
    name = f"test"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref53.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [53],
                "ref_link": ["https://doi.org/10.1021/ar500454v"],
                "reaction_type": ["oxidation"],
            }
        )
    )
df_ref53 = pd.concat(lst_df_ref53, ignore_index=True)

lst_df_ref54 = []
for idx, smi in enumerate(chemdraw_smiles_ref54):
    # print(smi)
    name = f"fluorination{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref54.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [54],
                "ref_link": ["https://doi.org/10.1021/ja5039819"],
                "reaction_type": ["H_abstraction_C-X"],
            }
        )
    )
df_ref54 = pd.concat(lst_df_ref54, ignore_index=True)

lst_df_ref56 = []
for idx, smi in enumerate(chemdraw_smiles_ref56):
    # print(smi)
    name = f"minisci{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref56.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [56],
                "ref_link": ["https://doi.org/10.1002/ange.201410432"],
                "reaction_type": ["H_abstraction_C-C"],
            }
        )
    )
df_ref56 = pd.concat(lst_df_ref56, ignore_index=True)

lst_df_ref59 = []
for idx, smi in enumerate(chemdraw_smiles_ref59):
    # print(smi)
    name = f"azidation{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref59.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [59],
                "ref_link": ["https://doi.org/10.1038/nature14127"],
                "reaction_type": ["H_abstraction_C-N"],
            }
        )
    )
df_ref59 = pd.concat(lst_df_ref59, ignore_index=True)

lst_df_ref123 = []
for idx, smi in enumerate(chemdraw_smiles_ref123):
    # print(smi)
    name = f"test"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref123.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [123],
                "ref_link": ["https://doi.org/10.1021/ja405471h"],
                "reaction_type": ["oxidation"],
            }
        )
    )
df_ref123 = pd.concat(lst_df_ref123, ignore_index=True)

lst_df_ref140 = []
for idx, smi in enumerate(chemdraw_smiles_ref140):
    # print(smi)
    name = f"test"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref140.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [140],
                "ref_link": ["https://doi.org/10.1002/anie.201407016"],
                "reaction_type": ["oxidation"],
            }
        )
    )
df_ref140 = pd.concat(lst_df_ref140, ignore_index=True)

lst_df_ref144 = []
for idx, smi in enumerate(chemdraw_smiles_ref144):
    # print(smi)
    name = f"ox{idx+4}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_ref144.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [144],
                "ref_link": ["https://doi.org/10.1021/ja00199a039"],
                "reaction_type": ["oxidation"],
            }
        )
    )
df_ref144 = pd.concat(lst_df_ref144, ignore_index=True)

lst_df_synthetic_applications = []
for idx, smi in enumerate(chemdraw_smiles_ref_synthetic_applications):
    # print(smi)
    name = f"appl{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_synthetic_applications.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [np.nan],
                "ref_link": ["10.1039/D1CS01169C"],
                "reaction_type": ["misc"],
            }
        )
    )
df_synthetic_applications = pd.concat(lst_df_synthetic_applications, ignore_index=True)

lst_df_alpha_funct = []
for idx, smi in enumerate(chemdraw_smiles_ref_alpha_funct):
    # print(smi)
    name = f"afunct{idx}"
    new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    (
        ref_mol_smiles_map,
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs_halator(
        name=name,
        smiles=new_smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    lst_df_alpha_funct.append(
        pd.DataFrame(
            {
                "names": [name],
                "chemdraw_smiles": [smi],
                "smiles": [new_smiles],
                "names_deprot": [lst_names_deprot],
                "smiles_deprot": [lst_smiles_deprot],
                "smiles_map_deprot": [lst_smiles_map_deprot],
                "atomsite_deprot": [lst_atomsite_deprot],
                "atomindex_deprot": [lst_atomindex_deprot],
                "ref_mol_smiles_map": [ref_mol_smiles_map],
                "ref": [np.nan],
                "ref_link": ["10.1039/D1CS01169C"],
                "reaction_type": ["misc"],
            }
        )
    )
df_alpha_funct = pd.concat(lst_df_alpha_funct, ignore_index=True)


merged_dict_reaction_sites = {
    **reaction_sites_ref38,
    **reaction_sites_ref43,
    **reaction_sites_ref44,
    **reaction_sites_ref47,
    **reaction_sites_ref48,
    **reaction_sites_ref51,
    **reaction_sites_ref52,
    **reaction_sites_ref53,
    **reaction_sites_ref54,
    **reaction_sites_ref56,
    **reaction_sites_ref59,
    **reaction_sites_ref123,
    **reaction_sites_ref140,
    **reaction_sites_ref144,
    **reaction_sites_synthetic_applications,
    **reaction_sites_alpha_funct,
}


df_merged_dict_reactionsites = pd.DataFrame.from_dict(
    merged_dict_reaction_sites, orient="index"
)
df_merged_dict_reactionsites.reset_index(inplace=True)
df_merged_dict_reactionsites.rename(columns={"index": "names"}, inplace=True)


df_merged_dataframes = pd.concat(
    [
        df_ref38,
        df_ref43,
        df_ref44,
        df_ref47,
        df_ref48,
        df_ref51,
        df_ref52,
        df_ref53,
        df_ref54,
        df_ref56,
        df_ref59,
        df_ref123,
        df_ref140,
        df_ref144,
        df_synthetic_applications,
        df_alpha_funct,
    ]
)

df_merged_dataframes.reset_index(inplace=True)

df_merged_dataframes = df_merged_dataframes.merge(
    pd.DataFrame(df_merged_dict_reactionsites), on="names"
)
df_merged_dataframes.to_pickle(
    "/groups/kemi/borup/HAlator/data/datasets/reaction_data_halator_20240611.pkl"
)
