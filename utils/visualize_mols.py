from pathlib import Path
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem import Draw
from collections import defaultdict
from io import BytesIO
from PIL import Image
import xml.etree.ElementTree as ET


def flatten_list(lst):
    return (
        [item for sublist in lst for item in sublist]
        if isinstance(lst[0], list) or isinstance(lst[0], tuple)
        else lst
    )


def draw_mol_highlight_qm_ml_halator(
    smiles,
    atomidx_min,
    atomindex_min_pred,
    atomidx_reaction_major,
    atomidx_reaction_minor,
    atomidx_reaction_steric,
    atomidx_reaction_electronic,
    legend="",
    img_size=(350, 300),
    draw_option="png",
    draw_legend=False,
    save_folder="",
    name="test",
):

    rdDepictor.SetPreferCoordGen(True)
    highlightatoms = defaultdict(list)
    atomrads = {}
    dict_color = {
        "green": (0.2, 1, 0.0, 1),
        "teal": (0.0, 0.5, 0.5, 1),
        "grey": (0.5, 0.5, 0.5, 1),
        "magenta": (1.0, 0.0, 1.0, 1),
        "blue": (0.0, 0.0, 1.0, 1),
        "orange": (1.0, 0.5, 0.0, 1),
        "black": (0.0, 0.0, 0.0, 0.7),
    }

    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDepictor.StraightenDepiction(rdkit_mol)

    if atomidx_reaction_minor is not None:
        atomidx_reaction_minor = flatten_list([atomidx_reaction_minor])
    if atomidx_reaction_steric is not None:
        atomidx_reaction_steric = flatten_list([atomidx_reaction_steric])
    if atomidx_reaction_electronic is not None:
        atomidx_reaction_electronic = flatten_list([atomidx_reaction_electronic])

    for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
        if atomidx_reaction_major is not None and atom_idx == atomidx_reaction_major:
            highlightatoms[atom_idx].append(dict_color["green"])
            atomrads[atom_idx] = 0.2
            # label = f"{atom_idx}"
        if atomidx_min is not None and atom_idx == atomidx_min:
            highlightatoms[atom_idx].append(dict_color["grey"])
            atomrads[atom_idx] = 0.2
            # label = f"{atom_idx}"
        if atomindex_min_pred is not None and atom_idx == atomindex_min_pred:
            highlightatoms[atom_idx].append(dict_color["teal"])
            atomrads[atom_idx] = 0.2
            # label = f"{atom_idx}"
        if atomidx_reaction_minor is not None and atom_idx in atomidx_reaction_minor:
            highlightatoms[atom_idx].append(dict_color["magenta"])
            atomrads[atom_idx] = 0.2
        if atomidx_reaction_steric is not None and atom_idx in atomidx_reaction_steric:
            highlightatoms[atom_idx].append(dict_color["blue"])
            atomrads[atom_idx] = 0.2
        if (
            atomidx_reaction_electronic is not None
            and atom_idx in atomidx_reaction_electronic
        ):
            highlightatoms[atom_idx].append(dict_color["orange"])
            atomrads[atom_idx] = 0.2
            # atom.SetProp("atomNote", label)

    if draw_option == "png":
        d2d = Draw.MolDraw2DCairo(img_size[0], img_size[1])
    elif draw_option == "svg":
        d2d = Draw.MolDraw2DSVG(img_size[0], img_size[1])
    dopts = d2d.drawOptions()
    dopts.addAtomIndices = True
    dopts.legendFontSize = 35  # legend font size
    dopts.atomHighlightsAreCircles = True
    dopts.fillHighlights = True
    dopts.annotationFontScale = 0.9
    dopts.centreMoleculesBeforeDrawing = True
    dopts.fixedScale = 0.95  # -1.0 #0.5
    # dopts.drawMolsSameScale = False
    mol = Draw.PrepareMolForDrawing(rdkit_mol)
    if draw_legend:
        d2d.DrawMoleculeWithHighlights(
            mol, legend, dict(highlightatoms), {}, dict(atomrads), {}
        )
    else:
        d2d.DrawMoleculeWithHighlights(
            mol, "", dict(highlightatoms), {}, dict(atomrads), {}
        )
    d2d.FinishDrawing()

    if draw_option == "png":
        bio = BytesIO(d2d.GetDrawingText())
        save_path = Path(f"{save_folder}/{name}.png")
        img = Image.open(bio)
        # img.save(
        #     save_path,
        #     dpi=(700, 600),
        #     transparent=False,
        #     facecolor="white",
        #     format="PNG",
        # )
        return (legend.split(" ")[0], Image.open(bio))
    elif draw_option == "svg":
        svg = d2d.GetDrawingText()
        svg.replace("svg:", "")
        # with open(f"{save_folder}/{name}.svg", "w") as f:
        #     f.write(svg)
        return (legend.split(" ")[0], svg)


def draw_multiple(mol_imgs, nPerRow=4, save=False, save_folder="", save_name=""):
    """
    draw multiple molecules in a grid
    optionally save the image. Note that it only saves pngs and not svgs
    """
    nRows = (len(mol_imgs) + nPerRow - 1) // nPerRow
    nCols = nPerRow
    imgSize = (mol_imgs[0].width * nCols, mol_imgs[0].height * nRows)
    res = Image.new("RGB", imgSize)

    image_grid = [(i % nPerRow, i // nPerRow) for i in range(nRows * nCols)]

    for i, (col, row) in enumerate(image_grid):
        if i < len(mol_imgs):
            mol_img = mol_imgs[i]
        else:
            mol_img = Image.new(
                "RGB", (mol_imgs[0].width, mol_imgs[0].height), color="white"
            )
        res.paste(mol_img, box=(col * mol_imgs[0].width, row * mol_imgs[0].height))

    if save:
        save_path = Path(f"{save_folder}/{save_name}.png")
        res.save(
            save_path,
            dpi=(700, 600),
            transparent=False,
            facecolor="white",
            format="PNG",
        )

    return res


def draw_multiple_svgs(mol_svgs, nPerRow=4, save=False, save_folder="", save_name=""):
    """
    Draw multiple molecules in a grid from SVG data
    optionally save the image. Note that it only saves SVGs.
    """
    nRows = (len(mol_svgs) + nPerRow - 1) // nPerRow
    nCols = nPerRow

    # Get width and height from the first SVG
    first_svg = mol_svgs[0][1]
    root = ET.fromstring(first_svg)
    svg_width = int(root.attrib["width"].replace("px", ""))
    svg_height = int(root.attrib["height"].replace("px", ""))

    # Create the final SVG document
    final_svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg")
    final_svg.set("width", str(svg_width * nCols))
    final_svg.set("height", str(svg_height * nRows))

    for i, (name, svg) in enumerate(mol_svgs):
        row = i // nPerRow
        col = i % nPerRow
        x_offset = col * svg_width
        y_offset = row * svg_height

        svg_tree = ET.fromstring(svg)
        # Wrap the SVG in a group with the transform attribute for positioning
        group = ET.Element("g", transform=f"translate({x_offset},{y_offset})")
        group.append(svg_tree)

        final_svg.append(group)

    final_svg_str = ET.tostring(final_svg, encoding="unicode")

    if save:
        save_path = f"{save_folder}/{save_name}.svg"
        with open(save_path, "w") as f:
            f.write(final_svg_str)

    return final_svg_str
