import sys
from pathlib import Path

home_directory = Path.cwd()
if home_directory.name != "HAlator":
    raise ValueError("Please run this script from the pKalculator directory")
sys.path.append(str(home_directory / "qm_halator"))
sys.path.append(str(home_directory / "smi2gcs"))
sys.path.append(str(home_directory / "utils"))

import argparse

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from io import BytesIO
from PIL import Image
from collections import defaultdict, OrderedDict
import lightgbm as lgb

from etl import get_cm5_desc_vector_halator
from modify_smiles import deprotonate, remove_Hs_halator
from visualize_mols import draw_mol_highlight_qm_ml_halator

# from DescriptorCreator.PrepAndCalcDescriptor import Generator


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Get ML predicted C-H hydricities of a molecule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--smiles",
        default="CC(=O)Cc1ccccc1",
        help="SMILES input for ML prediction of hydricities",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--name",
        default="comp2",
        help="The name of the molecule",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="model/final_reg_model_all_data_dart_default_nshells3.txt",
        help="Path to the model file to use for prediction",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--error",
        default=0.0,
        help="Identify the possible site of reaction within (number) kcal/mol of the lowest hydricity",
        type=float,
    )

    args = parser.parse_args()

    return args


# def draw_mol_highlight(
#     smiles,
#     lst_atomindex,
#     lst_pka_pred,
#     error=0.0,
#     legend="",
#     img_size=(350, 300),
#     draw_option="png",
#     draw_legend=False,
#     save_folder="",
#     name="test",
# ):

#     rdDepictor.SetPreferCoordGen(True)
#     highlightatoms = defaultdict(list)
#     atomrads = {}
#     dict_color = {
#         "green": (0.2, 1, 0.0, 1),
#         "teal": (0.0, 0.5, 0.5, 1),
#     }

#     rdkit_mol = Chem.MolFromSmiles(smiles)
#     rdDepictor.Compute2DCoords(rdkit_mol)
#     rdDepictor.StraightenDepiction(rdkit_mol)

#     dict_atomidx_pka = {
#         atom_index: pka
#         for atom_index, pka in zip(lst_atomindex, lst_pka_pred)
#         if abs(pka - min(lst_pka_pred)) <= error
#     }

#     sorted_dict_atomidx_pka = OrderedDict(
#         sorted(dict_atomidx_pka.items(), key=lambda x: x[1], reverse=False)
#     )

#     for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
#         if atom_idx in sorted_dict_atomidx_pka.keys():
#             highlightatoms[atom_idx].append(dict_color["teal"])
#             atomrads[atom_idx] = 0.2
#             label = f"{sorted_dict_atomidx_pka[atom_idx]:.2f}"
#             # atom.SetProp("atomNote", label)

#     if draw_option == "png":
#         d2d = Draw.MolDraw2DCairo(img_size[0], img_size[1])
#     elif draw_option == "svg":
#         d2d = Draw.MolDraw2DSVG(img_size[0], img_size[1])
#     dopts = d2d.drawOptions()
#     dopts.addAtomIndices = True
#     dopts.legendFontSize = 35  # legend font size
#     dopts.atomHighlightsAreCircles = True
#     dopts.fillHighlights = True
#     dopts.annotationFontScale = 0.9
#     dopts.centreMoleculesBeforeDrawing = True
#     dopts.fixedScale = 0.95  # -1.0 #0.5
#     # dopts.drawMolsSameScale = False
#     mol = Draw.PrepareMolForDrawing(rdkit_mol)
#     if draw_legend:
#         d2d.DrawMoleculeWithHighlights(
#             mol, legend, dict(highlightatoms), {}, dict(atomrads), {}
#         )
#     else:
#         d2d.DrawMoleculeWithHighlights(
#             mol, "", dict(highlightatoms), {}, dict(atomrads), {}
#         )
#     d2d.FinishDrawing()

#     if draw_option == "png":
#         bio = BytesIO(d2d.GetDrawingText())
#         save_path = Path(f"{save_folder}/{name}.png")
#         img = Image.open(bio)
#         img.save(
#             save_path,
#             dpi=(700, 600),
#             transparent=False,
#             facecolor="white",
#             format="PNG",
#         )
#         # return (legend.split(" ")[0], Image.open(bio))
#     elif draw_option == "svg":
#         svg = d2d.GetDrawingText()
#         svg.replace("svg:", "")
#         with open(f"{save_folder}/{name}.svg", "w") as f:
#             f.write(svg)
#         # return (legend.split(" ")[0], svg)


if __name__ == "__main__":
    args = get_args()
    smiles = args.smiles
    name = args.name
    model = args.model
    error = args.error
    model = Path.cwd() / model
    save_folder = Path.cwd() / "data/ml_predictions"
    if not save_folder.exists():
        save_folder.mkdir(exist_ok=True)

    if not model.exists():
        raise ValueError(
            f"Model file {model} does not exist. Check model path and try again"
        )
    reg_model_full = lgb.Booster(model_file=model)
    # reg_model_full = lgb.Booster(model_file='full_models/final_reg_model_all_data_dart_default_nshells3_20240607.txt')
    print("-" * 50)
    print(f"Loaded model: {model}")
    print("-" * 50)
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    # new code
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
        smiles=smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
        rxn="rm_hydride",
    )

    print(f"found {len(lst_atomindex_deprot)} C-H sites: {lst_atomindex_deprot}")

    (
        cm5_list,
        descriptor_vector,
        mapper_vector,
    ) = get_cm5_desc_vector_halator(
        smi_name=name,
        smiles=smiles,
        atom_sites=lst_atomindex_deprot,
        n_shells=3,
        benchmark=False,
    )

    ML_predicted_hydricities = list(reg_model_full.predict(descriptor_vector))
    print("ML predicted C-H hydricities: ")
    print(
        f"{[(atom_idx, round(hydricity, 2)) for atom_idx, hydricity in zip(lst_atomindex_deprot, ML_predicted_hydricities)]}"
    )

    atomidx_min_pred = lst_atomindex_deprot[
        ML_predicted_hydricities.index(min(ML_predicted_hydricities))
    ]

    dict_atomidx_hydricity = {
        atom_index: hydricity
        for atom_index, hydricity in zip(lst_atomindex_deprot, ML_predicted_hydricities)
        if abs(hydricity - min(ML_predicted_hydricities)) <= error
    }

    sorted_dict_atomidx_hydricity = OrderedDict(
        sorted(dict_atomidx_hydricity.items(), key=lambda x: x[1], reverse=False)
    )
    print(f"ML predicted C-H hydricities within error range of {error}:")
    for k, v in sorted_dict_atomidx_hydricity.items():
        print(f"Atom index: {k}, Hydricity: {v:.2f} kcal/mol")

    img_name, img_svg = draw_mol_highlight_qm_ml_halator(
        smiles=smiles,
        atomidx_min=None,
        atomindex_min_pred=list(sorted_dict_atomidx_hydricity.keys())[0],
        atomidx_reaction_major=None,
        atomidx_reaction_minor=None,
        atomidx_reaction_steric=None,
        atomidx_reaction_electronic=list(sorted_dict_atomidx_hydricity.keys())[1:],
        legend=f"{name}",
        img_size=(300, 300),
        draw_option="svg",
        draw_legend=False,
        save_folder="",
        name=name,
    )

    save_path = f"{save_folder}/{name}.svg"
    with open(save_path, "w") as f:
        f.write(img_svg)
