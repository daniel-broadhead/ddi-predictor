from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_fp(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Invalid SMILES (unparseable molecules) return a zero vector
    rather than raising, so batch processing over a large dataset
    doesn't halt on malformed source data. dtype is explicitly
    uint8 — fingerprint bits are only ever 0 or 1, and letting
    NumPy default to float64 here silently balloons memory usage
    (an 8x increase) once mixed with any zero-vector fallbacks.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)
