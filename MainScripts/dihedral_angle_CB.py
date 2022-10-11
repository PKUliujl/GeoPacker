

import Bio
import Bio.PDB
from Bio.PDB.vectors import calc_dihedral,calc_angle
from math import *
import numpy as np
import os
import warnings
import torch
warnings.filterwarnings('ignore')

standard_aa_names = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    ]




def dihedral_angle(inputPATH,name,pdbPATH,chainID, begins_num, ends_num):
    structure = Bio.PDB.PDBParser().get_structure(name,pdbPATH)

    for model in structure:
     for chain in model:
      if chain.id == chainID:
        delete_REScluster=[]
        for residue in chain:
            #print(residue.id)
            if residue.get_resname() not in standard_aa_names:
                delete_REScluster.append(residue.id)
        if delete_REScluster!=[]:
            #print(delete_REScluster)
            for delete_res in delete_REScluster:
                chain.detach_child(delete_res)

    for model in structure:
        for chain in model:
            if chain.id in chainID:
                chain_pair = []
                total_res = [i for i in chain]
                for residue1 in total_res[int(begins_num): int(ends_num)]:
                    for residue2 in chain: 
                        eachpair = []
                        if residue1.id != residue2.id:
                            O1N2 = calc_dihedral( residue1['O'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                            N1O2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector(),residue2['O'].get_vector())
                            eachpair.extend([sin(O1N2),cos(O1N2),sin(N1O2),cos(N1O2)])
                        else:
                            eachpair.extend([0,-1,0,-1])

                        ##trRosetta dihedral angles and planar angles
                        if residue1.id != residue2.id:
                            if residue1.resname !='GLY' and residue2.resname !='GLY':
                                CB1CB2 = calc_dihedral( residue1['CA'].get_vector(), residue1['CB'].get_vector(), residue2['CB'].get_vector(),residue2['CA'].get_vector())
                                N1_CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue1['CB'].get_vector(),residue2['CB'].get_vector())
                                CB1_N2 = calc_dihedral( residue1['CB'].get_vector(), residue2['CB'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                                ##planar angles
                                CA1CB2 = calc_angle(residue1['CA'].get_vector(), residue1['CB'].get_vector(), residue2['CB'].get_vector())
                                CB1CA2 = calc_angle(residue1['CB'].get_vector(), residue2['CB'].get_vector(), residue2['CA'].get_vector())
                                eachpair.extend([sin(CB1CB2),cos(CB1CB2),sin(N1_CB2),cos(N1_CB2),sin(CB1_N2),cos(CB1_N2), sin(CA1CB2),cos(CA1CB2),sin(CB1CA2),cos(CB1CA2)])
                            elif residue1.resname =='GLY' and residue2.resname !='GLY':
                                CB1CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CB'].get_vector(),residue2['CA'].get_vector())
                                N1_CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['C'].get_vector(), residue1['CA'].get_vector(),residue2['CB'].get_vector())
                                CB1_N2 = calc_dihedral( residue1['CA'].get_vector(), residue2['CB'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                                CA1CB2 = calc_angle(residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CB'].get_vector())
                                CB1CA2 = calc_angle(residue1['CA'].get_vector(), residue2['CB'].get_vector(), residue2['CA'].get_vector())
                                eachpair.extend([sin(CB1CB2),cos(CB1CB2),sin(N1_CB2),cos(N1_CB2),sin(CB1_N2),cos(CB1_N2),sin(CA1CB2),cos(CA1CB2),sin(CB1CA2),cos(CB1CA2)])
                            elif residue1.resname !='GLY' and residue2.resname =='GLY':
                                CB1CB2 = calc_dihedral( residue1['CA'].get_vector(), residue1['CB'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                                N1_CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue1['CB'].get_vector(),residue2['CA'].get_vector())
                                CB1_N2 = calc_dihedral( residue1['CB'].get_vector(), residue2['CA'].get_vector(), residue2['C'].get_vector(),residue2['N'].get_vector())
                                CA1CB2 = calc_angle(residue1['CA'].get_vector(), residue1['CB'].get_vector(), residue2['CA'].get_vector())
                                CB1CA2 = calc_angle(residue1['CB'].get_vector(), residue2['CA'].get_vector(), residue2['N'].get_vector())
                                eachpair.extend([sin(CB1CB2),cos(CB1CB2),sin(N1_CB2),cos(N1_CB2),sin(CB1_N2),cos(CB1_N2),sin(CA1CB2),cos(CA1CB2),sin(CB1CA2),cos(CB1CA2)])
                            else:
                                CB1CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                                N1_CB2 = calc_dihedral( residue1['N'].get_vector(), residue1['C'].get_vector(), residue1['CA'].get_vector(),residue2['CA'].get_vector())
                                CB1_N2 = calc_dihedral( residue1['CA'].get_vector(), residue2['CA'].get_vector(), residue2['C'].get_vector(),residue2['N'].get_vector())
                                CA1CB2 = calc_angle(residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector())
                                CB1CA2 = calc_angle(residue1['CA'].get_vector(), residue2['CA'].get_vector(), residue2['N'].get_vector())
                                eachpair.extend([sin(CB1CB2),cos(CB1CB2),sin(N1_CB2),cos(N1_CB2),sin(CB1_N2),cos(CB1_N2),sin(CA1CB2),cos(CA1CB2),sin(CB1CA2),cos(CB1CA2)])
                        else:
                            eachpair.extend([0,-1,0,-1,0,-1,0,-1,0,-1])
                            

                        chain_pair.append(eachpair)
                chain_pair = np.array(chain_pair).reshape( ( int(ends_num)-int(begins_num),len(chain),14) )
                #print('-----:',chain_pair.max())
                #np.save( os.path.join(inputPATH, name + '_' + chainID+'_orientation') ,chain_pair) 
                #print(pdbPATH,pdbPATH.split('.')[0] +'dihedral_CB')
                print(inputPATH,name,pdbPATH,chainID, int(begins_num), int(ends_num),'Finished')
                return torch.from_numpy(chain_pair)
            else:
                break

'''
import pandas as pd

data_list = pd.read_csv('whole_list' ,header=None,sep='\s+')

from sys import argv

for i in range(int(argv[1]),int(argv[2]) ): #(7000,len(data_list)):
    if data_list.iloc[i,0]+'_'+data_list.iloc[i,1]+'.npy' not in os.listdir('dihedral_CB_perturbe/'):
        print( i )
        diang = dihedral_angle( data_list.iloc[i,0], '/lustre2/lhlai_pkuhpc/liujl/RD_LR/pdb_chain_perturbe/'+data_list.iloc[i,0]+'_'+data_list.iloc[i,1]+'.pdb' ,data_list.iloc[i,1] )
        np.save('dihedral_CB_perturbe/'+data_list.iloc[i,0]+'_'+data_list.iloc[i,1],diang)
'''
