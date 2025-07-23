"""
Li Vacancy Creation Script

Delete the first Li atom in every CIF file in the current directory 
and save as *_LiVac0.cif.

Author: lunazhang
Date: 2023

Usage:
    conda activate cgcnn
    cd vacancy_data
    python make_li_vac.py
"""

import glob
import os
from pymatgen.core import Structure

def main():
    for cif in glob.glob("*.cif"):
        if cif.endswith("_LiVac0.cif"):
            continue  # skip already processed
        try:
            struct = Structure.from_file(cif)
        except Exception as e:
            print(f"Failed to read {cif}: {e}")
            continue

        li_indices = [i for i, site in enumerate(struct) if site.specie.symbol == "Li"]
        if not li_indices:
            print(f"No Li found in {cif}, skipping.")
            continue

        struct.remove_sites([li_indices[0]])
        out_file = cif.replace(".cif", "_LiVac0.cif")
        struct.to(fmt="cif", filename=out_file)
        print(f"Created {out_file}")

if __name__ == "__main__":
    main()