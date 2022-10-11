# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:17:36 2021

@author: Administrator
"""

import Bio
import Bio.PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.NACCESS import run_naccess
import pandas as pd 
from math import *
import math
import numpy as np
import torch


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


## 20 standard anino acids

AA = {'A':0, 
      'C':1,
      'D':2,
      'E':3,
      'F':4,
      'G':5,
      'H':6,
      'I':7,
      'K':8,
      'L':9,
      'M':10,
      'N':11,
      'P':12,
      'Q':13,
      'R':14,
      'S':15,
      'T':16,
      'V':17,
      'W':18,
      'Y':19}


def distance_diff(A,B):
    square=0
    for i in range(3):
        square = square + (A[i]-B[i])**2
    return math.sqrt(square)



#structure = Bio.PDB.PDBParser().get_structure('1a21','1a21.pdb')
#structure = Bio.PDB.PDBParser().get_structure('1a12','1a12.pdb')structure = Bio.PDB.PDBParser(QUIET=True).get_structure('2d69','2d69.pdb')
#structure = Bio.PDB.PDBParser(QUIET=True).get_structure('2qh7','2qh7.pdb')


##DSSP 8 states reduce to 3 states
## H,G -> H; E,B ->E ; else ->C
def eight2three(ss):
    if ss=='H':
        return [1,0,0,0,0,0,0,0]
    elif ss=='G':
        return [0,1,0,0,0,0,0,0]
    elif ss=='I':
        return [0,0,1,0,0,0,0,0]
    elif ss=='E':
        return [0,0,0,1,0,0,0,0]
    elif ss=='B':
        return [0,0,0,0,1,0,0,0]
    elif ss=='T':
        return [0,0,0,0,0,1,0,0]
    elif ss=='S':
        return [0,0,0,0,0,0,1,0]
    else:
        return [0,0,0,0,0,0,0,1]



def preprocess_complex(inputPATH,pdbname, inputfile, chainID, begins_num, ends_num ):
    structure = Bio.PDB.PDBParser().get_structure(pdbname,inputfile)
    assert len(structure) ==1, 'There are multiple models in the input pdb file. Only one model is accepted.'
        
    #delete irregular residue atoms
    for model in structure:
        for chain in model:
          if chain.id in chainID:
            delete_REScluster=[]
            for residue in chain:
                #print(residue.id)
                if residue.get_resname() not in standard_aa_names:
                    delete_REScluster.append(residue.id)
            if delete_REScluster!=[]:
                #print(delete_REScluster)
                for delete_res in delete_REScluster:
                    chain.detach_child(delete_res)
                    
    #neighbor search    
    atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')
    ns = Bio.PDB.NeighborSearch(atoms)

    for model in structure:  ##only consider 1 model in the PDBfile
        dssp = DSSP(model,inputfile)
        #naccess = run_naccess(model,'2qh7.pdb',temp_path=r"G:\TEMP")
        phi_psi_list=[]
        sequence_list=''
        residue_list=[]
        CHAINID=''
        for chain in model:
          if chain.id in chainID:  
            #print('\n-------\n%s %s:\n'%(structure.id,chain.id))
            poly=Bio.PDB.Polypeptide.Polypeptide(chain)
            phi_psi_list.extend( poly.get_phi_psi_list() )
            sequence_list += poly.get_sequence()
            if int(begins_num)==0:
                print('\n-------\n%s %s:\n'%(structure.id,chain.id))
                print('Sequence: ',poly.get_sequence())
            residue_list.extend([residue.id for residue in chain])
            CHAINID +=chain.id*len(poly.get_sequence())
        #print()
        matrix = [[i,chainid,j,l,dssp[(chainid,model[chainid][int(l)].id)][2],n,p] for i,(j,chainid,(k,l,m),(n,p)) in enumerate(zip(sequence_list,CHAINID,residue_list,phi_psi_list))  ]
        #print(matrix)
        matrix = pd.DataFrame(matrix)
        #matrix.to_csv(argv[4]+argv[1]+'_' + argv[3]+'_list',header=None,sep=' ',index=False)
        #print (matrix) 
        node_feature = []
        node_category = []
        edge_attributes = []
        edge_index =[]
        for i in range(int(begins_num),int(ends_num) ):
            each_node_feature = []
            ####first four columns are sin *cos (phi psi)
            if pd.isna(matrix.iloc[i,-2]):
                each_node_feature.extend( [ 0,0,sin(matrix.iloc[i,-1]),cos(matrix.iloc[i,-1]),   0,0,0,sin(matrix.iloc[i,-1]*2),sin(matrix.iloc[i,-1]*3),sin(matrix.iloc[i,-1]*4) ]     )
            elif pd.isna(matrix.iloc[i,-1]):
                each_node_feature.extend( [ sin(matrix.iloc[i,-2]),cos(matrix.iloc[i,-2]),0,0,   sin(matrix.iloc[i,-2]*2),sin(matrix.iloc[i,-2]*3),sin(matrix.iloc[i,-2]*4),0,0,0 ]      )
            else:
                each_node_feature.extend( [ sin(matrix.iloc[i,-2]),cos(matrix.iloc[i,-2]),sin(matrix.iloc[i,-1]),cos(matrix.iloc[i,-1]),       sin(matrix.iloc[i,-2]*2),sin(matrix.iloc[i,-2]*3),sin(matrix.iloc[i,-2]*4),       sin(matrix.iloc[i,-1]*2),sin(matrix.iloc[i,-1]*3),sin(matrix.iloc[i,-1]*4)  ]     )
            ###The next three columns are 3 states of DSSP
            each_node_feature.extend( eight2three( matrix.iloc[i,4] )  )
            node_category.append(AA[matrix.iloc[i,2]])
            ##local coordinatate 
            CA_coord = model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CA'].coord  #model[matrix .iloc[i,1]]
            C_coord = model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['C'].coord
            N_coord = model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['N'].coord
            O_coord = model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['O'].coord
            CA_C = C_coord - CA_coord    ## norm   np.linalg.norm( CA_C )
            CA_N = N_coord - CA_coord 
            CA_O = O_coord - CA_coord
            orthognal = np.cross(CA_C, CA_N)
            #each_node_feature.extend(  CA_C.tolist() )
            #each_node_feature.extend( CA_N.tolist() )
            #each_node_feature.extend( orthognal.tolist())
            node_feature.append(each_node_feature )
            NA = ns.search(model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CA'].coord, 10)  ##Neighbor Atoms
            #print(i,matrix.iloc[i,3],[atom.get_parent().id[1] for atom in NA if atom.id=='CA'])
            All_CA = [atom for atom in NA if atom.id=='CA']   # atom.get_parent().id[1]
            for CA in All_CA: 
                if (CA.coord != model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CA'].coord).any() and CA.get_parent().get_parent().id in chainID:
                    each_edge_attributes = []
                    each_edge_index = []
                    #print(model[matrix .iloc[i,1]][int(matrix .iloc[i,3])].id[1], CA.get_parent().id[1] )
                    CA_CA_dis = np.linalg.norm( CA.coord - CA_coord )  #distance between two CA
                    CA_CA_orientation = [np.dot( (CA.coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]
                    ## CA -> C
                    CA_C_dis = np.linalg.norm( CA.get_parent()['C'].coord - CA_coord )
                    CA_C_orientation = [np.dot( CA.get_parent()['C'].coord - CA_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['C'].coord - CA_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['C'].coord - CA_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## CA -> N
                    CA_N_dis = np.linalg.norm( CA.get_parent()['N'].coord - CA_coord )
                    CA_N_orientation = [np.dot( CA.get_parent()['N'].coord - CA_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['N'].coord - CA_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['N'].coord - CA_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## CA_O 
                    CA_O_dis = np.linalg.norm( CA.get_parent()['O'].coord - CA_coord )
                    CA_O_orientation = [np.dot( CA.get_parent()['O'].coord - CA_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['O'].coord - CA_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['O'].coord - CA_coord , orthognal )/np.linalg.norm( orthognal )]
                    ##CA_CB
                    if CA.get_parent().resname !='GLY':
                        CA_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - CA_coord )
                        CA_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - CA_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['CB'].coord - CA_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - CA_coord , orthognal )/np.linalg.norm( orthognal )]
                    else:
                        CA_CB_dis = CA_CA_dis
                        CA_CB_orientation = CA_CA_orientation
                    ## O -> N, where O is from the central residue and N is from the neighbor residue
                    O_N_dis = np.linalg.norm( CA.get_parent()['N'].coord - O_coord )
                    O_N_orientation = [np.dot( CA.get_parent()['N'].coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['N'].coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['N'].coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## O ->C
                    O_C_dis = np.linalg.norm( CA.get_parent()['C'].coord - O_coord )
                    O_C_orientation = [np.dot( CA.get_parent()['C'].coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['C'].coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['C'].coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## O ->O
                    O_O_dis = np.linalg.norm( CA.get_parent()['O'].coord - O_coord )
                    O_O_orientation = [np.dot( CA.get_parent()['O'].coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['O'].coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['O'].coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## O ->CA
                    O_CA_dis = np.linalg.norm( CA.coord - O_coord )
                    O_CA_orientation = [np.dot( CA.coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## o ->CB
                    if CA.get_parent().resname !='GLY':
                        O_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - O_coord )
                        O_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['CB'].coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
                    else:
                        O_CB_dis = O_CA_dis
                        O_CB_orientation = O_CA_orientation
                    ## N -> O, where N is from the central residue and O is from the neighbor residue
                    N_O_dis = np.linalg.norm( CA.get_parent()['O'].coord - N_coord )
                    N_O_orientation = [np.dot( CA.get_parent()['O'].coord - N_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['O'].coord - N_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['O'].coord - N_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## N -> C
                    N_C_dis = np.linalg.norm( CA.get_parent()['C'].coord - N_coord )
                    N_C_orientation = [np.dot( CA.get_parent()['C'].coord - N_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['C'].coord - N_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['C'].coord - N_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## N -> N
                    N_N_dis = np.linalg.norm( CA.get_parent()['N'].coord - N_coord )
                    N_N_orientation = [np.dot( CA.get_parent()['N'].coord - N_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['N'].coord - N_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['N'].coord - N_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## N -> CA
                    N_CA_dis = np.linalg.norm( CA.coord - N_coord )
                    N_CA_orientation = [np.dot( CA.coord - N_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.coord - N_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.coord - N_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## N ->CB
                    if CA.get_parent().resname !='GLY':
                        N_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - N_coord )
                        N_CB_orientation= [np.dot( CA.get_parent()['CB'].coord - N_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['CB'].coord - N_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - N_coord , orthognal )/np.linalg.norm( orthognal )]
                    else:
                        N_CB_dis = N_CA_dis
                        N_CB_orientation = N_CA_orientation
                        
                    ## C -> C
                    C_C_dis = np.linalg.norm( CA.get_parent()['C'].coord - C_coord )
                    C_C_orientation = [np.dot( CA.get_parent()['C'].coord - C_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['C'].coord - C_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['C'].coord - C_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## C -> N
                    C_N_dis = np.linalg.norm( CA.get_parent()['N'].coord - C_coord )
                    C_N_orientation = [np.dot( CA.get_parent()['N'].coord - C_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['N'].coord - C_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['N'].coord - C_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## C -> O
                    C_O_dis = np.linalg.norm( CA.get_parent()['O'].coord - C_coord )
                    C_O_orientation = [np.dot( CA.get_parent()['O'].coord - C_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['O'].coord - C_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['O'].coord - C_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## C -> CA
                    C_CA_dis = np.linalg.norm( CA.coord - C_coord )
                    C_CA_orientation = [np.dot( CA.coord - C_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.coord - C_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.coord - C_coord , orthognal )/np.linalg.norm( orthognal )]
                    ## C ->CB
                    if CA.get_parent().resname !='GLY':
                        C_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - C_coord )
                        C_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - C_coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['CB'].coord - C_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - C_coord , orthognal )/np.linalg.norm( orthognal )]
                    else:
                        C_CB_dis = C_CA_dis
                        C_CB_orientation = C_CA_orientation
                    
                    ## CB ->CB
                    if model[matrix .iloc[i,1]][int(matrix .iloc[i,3])].resname !='GLY':
                        CB_C_dis = np.linalg.norm( CA.get_parent()['C'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord )
                        CB_C_orientation = [np.dot( CA.get_parent()['C'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['C'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['C'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , orthognal )/np.linalg.norm( orthognal )]
                        CB_N_dis = np.linalg.norm( CA.get_parent()['N'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord )
                        CB_N_orientation =  [np.dot( CA.get_parent()['N'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['N'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['N'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , orthognal )/np.linalg.norm( orthognal )]
                        CB_O_dis = np.linalg.norm( CA.get_parent()['O'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord )
                        CB_O_orientation = [np.dot( CA.get_parent()['O'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['O'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['O'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , orthognal )/np.linalg.norm( orthognal )]
                        CB_CA_dis = np.linalg.norm( CA.get_parent()['CA'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord )
                        CB_CA_orientation = [np.dot( CA.get_parent()['CA'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['CA'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CA'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , orthognal )/np.linalg.norm( orthognal )]
                        if CA.get_parent().resname !='GLY':
                            CB_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord )
                            CB_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_C )/np.linalg.norm( CA_C ),  np.dot( CA.get_parent()['CB'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - model[matrix .iloc[i,1]][int(matrix .iloc[i,3])]['CB'].coord , orthognal )/np.linalg.norm( orthognal )]
                        else:
                            CB_CB_dis = CB_CA_dis
                            CB_CB_orientation = CB_CA_orientation
                    else:
                        CB_C_dis = CA_C_dis #np.linalg.norm( CA.get_parent()['C'].coord - CA_coord )
                        CB_C_orientation = CA_C_orientation #[np.dot( (CA.get_parent()['C'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['C'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['C'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]
                        CB_N_dis = CA_N_dis #np.linalg.norm( CA.get_parent()['N'].coord - CA_coord )
                        CB_N_orientation = CA_N_orientation #[np.dot( (CA.get_parent()['N'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['N'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['N'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]    
                        CB_O_dis = CA_O_dis #np.linalg.norm( CA.get_parent()['O'].coord - CA_coord )
                        CB_O_orientation = CA_O_orientation #[np.dot( (CA.get_parent()['O'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['O'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['O'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]    
                        CB_CA_dis = CA_CA_dis
                        CB_CA_orientation = CA_CA_orientation
                        if CA.get_parent().resname !='GLY':
                            CB_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - CA_coord )
                            CB_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - CA_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['CB'].coord - CA_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - CA_coord , orthognal )/np.linalg.norm( orthognal )]
                        else:
                            CB_CB_dis = CA_CA_dis
                            CB_CB_orientation = CA_CA_orientation
                            
                    each_edge_attributes.extend([CA_CA_dis,1/CA_CA_dis**2,1/CA_CA_dis**4,1/CA_CA_dis**6])
                    each_edge_attributes.extend(CA_CA_orientation)
                    each_edge_attributes.extend([CA_C_dis,1/CA_C_dis**2,1/CA_C_dis**4,1/CA_C_dis**6])
                    each_edge_attributes.extend(CA_C_orientation)
                    each_edge_attributes.extend([CA_O_dis,1/CA_O_dis**2,1/CA_O_dis**4,1/CA_O_dis**6])
                    each_edge_attributes.extend(CA_O_orientation)
                    each_edge_attributes.extend([CA_N_dis,1/CA_N_dis**2,1/CA_N_dis**4,1/CA_N_dis**6])
                    each_edge_attributes.extend(CA_N_orientation)
                    each_edge_attributes.extend([CA_CB_dis,1/CA_CB_dis**2,1/CA_CB_dis**4,1/CA_CB_dis**6])
                    each_edge_attributes.extend(CA_CB_orientation)
                    
                    each_edge_attributes.extend([O_N_dis,1/O_N_dis**2,1/O_N_dis**4,1/O_N_dis**6])
                    each_edge_attributes.extend(O_N_orientation)
                    each_edge_attributes.extend([O_C_dis,1/O_C_dis**2,1/O_C_dis**4,1/O_C_dis**6])
                    each_edge_attributes.extend(O_C_orientation)
                    each_edge_attributes.extend([O_CA_dis,1/O_CA_dis**2,1/O_CA_dis**4,1/O_CA_dis**6])
                    each_edge_attributes.extend(O_CA_orientation)
                    each_edge_attributes.extend([O_O_dis,1/O_O_dis**2,1/O_O_dis**4,1/O_O_dis**6])
                    each_edge_attributes.extend(O_O_orientation)
                    each_edge_attributes.extend([O_CB_dis,1/O_CB_dis**2,1/O_CB_dis**4,1/O_CB_dis**6])
                    each_edge_attributes.extend(O_CB_orientation)
    
                    each_edge_attributes.extend([N_O_dis,1/N_O_dis**2,1/N_O_dis**4,1/N_O_dis**6])
                    each_edge_attributes.extend(N_O_orientation)
                    each_edge_attributes.extend([N_C_dis,1/N_C_dis**2,1/N_C_dis**4,1/N_C_dis**6])
                    each_edge_attributes.extend(N_C_orientation)
                    each_edge_attributes.extend([N_CA_dis,1/N_CA_dis**2,1/N_CA_dis**4,1/N_CA_dis**6])
                    each_edge_attributes.extend(N_CA_orientation)
                    each_edge_attributes.extend([N_N_dis,1/N_N_dis**2,1/N_N_dis**4,1/N_N_dis**6])
                    each_edge_attributes.extend(N_N_orientation)
                    each_edge_attributes.extend([N_CB_dis,1/N_CB_dis**2,1/N_CB_dis**4,1/N_CB_dis**6])
                    each_edge_attributes.extend(N_CB_orientation)
                    
                    each_edge_attributes.extend([C_O_dis,1/C_O_dis**2,1/C_O_dis**4,1/C_O_dis**6])
                    each_edge_attributes.extend(C_O_orientation)
                    each_edge_attributes.extend([C_N_dis,1/C_N_dis**2,1/C_N_dis**4,1/C_N_dis**6])
                    each_edge_attributes.extend(C_N_orientation)
                    each_edge_attributes.extend([C_C_dis,1/C_C_dis**2,1/C_C_dis**4,1/C_C_dis**6])
                    each_edge_attributes.extend(C_C_orientation)
                    each_edge_attributes.extend([C_CA_dis,1/C_CA_dis**2,1/C_CA_dis**4,1/C_CA_dis**6])
                    each_edge_attributes.extend(C_CA_orientation)
                    each_edge_attributes.extend([C_CB_dis,1/C_CB_dis**2,1/C_CB_dis**4,1/C_CB_dis**6])
                    each_edge_attributes.extend(C_CB_orientation)
    
                    each_edge_attributes.extend([CB_O_dis,1/CB_O_dis**2,1/CB_O_dis**4,1/CB_O_dis**6])
                    each_edge_attributes.extend(CB_O_orientation)
                    each_edge_attributes.extend([CB_N_dis,1/CB_N_dis**2,1/CB_N_dis**4,1/CB_N_dis**6])
                    each_edge_attributes.extend(CB_N_orientation)
                    each_edge_attributes.extend([CB_C_dis,1/CB_C_dis**2,1/CB_C_dis**4,1/CB_C_dis**6])
                    each_edge_attributes.extend(CB_C_orientation)
                    each_edge_attributes.extend([CB_CA_dis,1/CB_CA_dis**2,1/CB_CA_dis**4,1/CB_CA_dis**6])
                    each_edge_attributes.extend(CB_CA_orientation)
                    each_edge_attributes.extend([CB_CB_dis,1/CB_CB_dis**2,1/CB_CB_dis**4,1/CB_CB_dis**6])
                    each_edge_attributes.extend(CB_CB_orientation)
                    
    
                    edge_attributes.append(each_edge_attributes)
                    ##beginnode to endnode 
                    endnode = matrix[ (matrix.iloc[:,3] == CA.get_parent().id[1]) & (matrix.iloc[:,1] == CA.get_parent().get_parent().id)   ].iloc[0,0]
                    each_edge_index.extend([i,endnode ])
                    edge_index.append(each_edge_index)
        #print( node_feature )
        #print(pd.DataFrame(node_feature).dtypes)
        #np.savetxt(os.path.join(inputPATH, pdbname + '_' + chainID+'_nodefeature'), np.array(node_feature) )             
        #print( node_feature )
        #np.savetxt(os.path.join(inputPATH, pdbname + '_' + chainID+'_nodefeature'), np.array(node_category) )
        #print( node_category )
        #np.savetxt(os.path.join(inputPATH, pdbname + '_' + chainID+'_nodefeature'), np.array(edge_attributes) )
        #print( edge_attributes )
        #np.savetxt(os.path.join(inputPATH, pdbname + '_' + chainID+'_nodefeature'), np.array(edge_index) )  
        #print ( edge_index )
        print("graph: ",int(begins_num), int(ends_num),end='\r')
        return torch.from_numpy(np.array(node_feature)), torch.from_numpy(np.array(node_category)).long(), torch.from_numpy(np.array(edge_attributes)), torch.from_numpy(np.array(edge_index).T).long()
    
                            
    
         
