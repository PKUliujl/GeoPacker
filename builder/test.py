
import numpy as np    
from myclass import Atoms, Residues,Myio
from buildprotein import RebuildStructure


def builder(inputpdbfile,chainID, torsions, outputpdbfile,seq_tobe_designed=None):
    atomsData_real = Myio.readPDB( inputpdbfile ,chainID )
    atomsData_mc = RebuildStructure.extractmc(atomsData_real = atomsData_real, seq_tobe_designed = seq_tobe_designed)
    residuesData_mc = Residues.getResidueData(atomsData_mc) 

    num_atoms = sum([i.num_side_chain_atoms for i in residuesData_mc]) + 5*len(residuesData_mc)

    geosData = RebuildStructure.getGeosData(residuesData_mc)

    residuesData_mc = RebuildStructure.rebuild_cb(residuesData_mc, geosData)

    init_atoms_matrix = np.zeros((num_atoms, 3)).astype(np.float32) 
    init_atoms_matrix  = RebuildStructure.make_atoms_matrix(residuesData_mc, init_atoms_matrix)

    atoms_matrix, atoms_matrix_name = RebuildStructure.rebuild( torsions, residuesData_mc, geosData, init_atoms_matrix)
    return Myio.outputPDB(residuesData_mc, atoms_matrix, atoms_matrix_name, outputpdbfile)

