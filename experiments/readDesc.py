import os
import numpy as np

import ase.io
from ase.calculators.espresso import Espresso
from ase.io.espresso import write_espresso_in
from ase.io import extxyz
from ase.atoms import Atoms

from pathlib import Path
import re
import glob
from natsort import natsorted, ns

import itertools
import random
import math
import struct

###################################################
# Read/Get descriptors from fitpod calculation.
######################################################

fileName = glob.glob("localdescriptors_config*.bin", recursive=False)
## Warning: Use natural sort for the filenames, because python globs in the order 1, 10, 100, ...
## instead of 1, 2, ... so the xyz configurations won't match with the in files tat contain the descriptors.
fileName = natsorted(fileName, key=lambda y: y.lower())

D = []

for fileDesc in fileName:

    numbers = []

    with open(fileDesc, mode="rb") as f:
        while (byte := f.read(8)):
            (number, ) = struct.unpack('d', byte)
            numbers.append(number)
    
    ## First float is the number of atoms
    Natoms = int(numbers.pop(0))
    ## Second float is the number of descriptors (per-atom)
    Mdesc = int(numbers.pop(0))
    
    ## For per-atom Descriptors, the file contains: Mdesc floats for the 1st atom, Mdesc for the second, etc.
    ## Note: For multi-element systems the per-atom descriptors are different for each element type.
    ## Keep track of the atom types for the configurations provided in the xyz files.
    D.append( np.array(numbers).reshape(int(Mdesc), int(Natoms)).T )
