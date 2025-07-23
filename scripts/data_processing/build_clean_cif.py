"""
CIF Structure Processing Script

Script to process CIF files by removing Li atoms and creating vacancy structures.

Author: lunazhang
Date: 2023
"""

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import glob, os

for raw in glob.glob("vacancy_data/sd_*.cif"):
    s = Structure.from_file(raw)
    li_idx = [i for i,site in enumerate(s) if site.specie.symbol == "Li"][0]
    s.remove_sites([li_idx])
    # Set all occupancies to 1 / 把所有占位设为 1
    for site in s:
        site.occu = 1
    out = raw.replace("sd_", "").replace(".cif", "_LiVac0.cif")
    CifWriter(s).write_file(out)
    print("Generated:", out)
