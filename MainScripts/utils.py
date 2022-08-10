

import numpy as np
import os,Bio
import Bio.PDB
from Bio.PDB.vectors import calc_dihedral
from math import *
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
import torch

import warnings
warnings.filterwarnings('ignore')

from single import *

standard_aa_names = {
                   "ALA":0,
                   "CYS":1,
                   "ASP":2,
                   "GLU":3,
                   "PHE":4,
                   "GLY":5,
                   "HIS":6,
                   "ILE":7,
                   "LYS":8,
                   "LEU":9,
                   "MET":10,
                   "ASN":11,
                   "PRO":12,
                   "GLN":13,
                   "ARG":14,
                   "SER":15,
                   "THR":16,
                   "VAL":17,
                   "TRP":18,
                   "TYR":19,
                   }

AA_single = {'A':0,
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
      'Y':19
      }


standard_aa_names_r = dict( [val,key ] for key,val in standard_aa_names.items() )


standard_a3_names = [
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


Rotamer_number = {0:0,
                 1:1,
                 2:2,
                 3:3,
                 4:2,
                 5:0,
                 6:2,
                 7:2,
                 8:4,
                 9:2,
                 10:3,
                 11:2,
                 12:1,
                 13:3,
                 14:4,
                 15:1,
                 16:1,
                 17:1,
                 18:2,
                 19:2
                }

def dihedral_angle(inputPATH, name,pdbPATH,chainID):
    structure = Bio.PDB.PDBParser().get_structure(name,pdbPATH)

    for model in structure:
     for chain in model:
      if chain.id in chainID:
        delete_REScluster=[]
        for residue in chain:
            #print(residue.id)
            if residue.get_resname() not in standard_a3_names:
                delete_REScluster.append(residue.id)
        if delete_REScluster!=[]:
            #print(delete_REScluster)
            for delete_res in delete_REScluster:
                chain.detach_child(delete_res)
    
    for model in structure:
        res = []
        for chain in model:
            if chain.id in chainID:
                for residue in chain:
                    res.append(residue)
        
        chain_pair = []
        for residue1 in res:
            for residue2 in res: 
                eachpair = []
                if residue1.id != residue2.id or residue1.get_parent().id != residue2.get_parent().id:
                    #print(residue1.id,residue2.id,chain.id)
                    O1N2 = calc_dihedral( residue1['O'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector(),residue2['N'].get_vector())
                    N1O2 = calc_dihedral( residue1['N'].get_vector(), residue1['CA'].get_vector(), residue2['CA'].get_vector(),residue2['O'].get_vector())
                    eachpair.extend([sin(O1N2),cos(O1N2),sin(N1O2),cos(N1O2)])
                else:
                    eachpair.extend([0,1,0,1])
                chain_pair.append(eachpair)
        chain_pair = np.array(chain_pair).reshape( (len(res),len(res),4) )
        #print('-----:',chain_pair.max())
        np.save( os.path.join(inputPATH, name + '_' + chainID+'_orientation') ,chain_pair)
        return torch.from_numpy(chain_pair)



def f2dgenerate(path,name,pdbname,chainID):
    file = open(os.path.join(path, name), 'r')
    # file = gzip.open(os.path.join('chains', '1a0tP.gz'), 'r')
    num = -9999
    trans = []
    x = 0.0
    y = 0.0
    z = 0.0
    flag = 0
    for row in file:
        if row[:3] == 'END':
            break
        elif row[:4]=='ATOM' and row[21] in chainID:
            now = int(row[22:26])
            # print(str(row[11:17]))
            # print(float(row[26:38]))
            # print(float(row[38:46]))
            # print(float(row[46:54]))
            if (num != -9999 and num != now):
                trans.append([x, y, z])
                flag = 0
            num = now
            if (row[11:17] == '  CB  '):
                x = float(row[27:38])
                y = float(row[38:46])
                z = float(row[46:54])
                flag = 1
            if (row[11:17] == '  CA  ' and flag == 0):
                x = float(row[27:38])
                y = float(row[38:46])
                z = float(row[46:54])
    trans.append([x, y, z])
    Trans = np.array(trans)

    y = []
    for i in range(Trans.__len__()):
        y.append(Trans)
    y = np.array(y)
    # print(y)
    x = y.transpose(1, 0, 2)
    # print(x)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)

    a = 2 / (1 + a / 4)
    for i in range(len(a)):
        a[i, i] = 1
    np.save( os.path.join(path, pdbname + '_' + chainID+'_dismap') ,a)
    return torch.from_numpy(a)


def read_seqfile(filename):
        f = open(filename)
        rows = f.readlines()
        f.close()
        if rows[0][0] !='>':
            print('Wrong FASTA format for input to be designed sequence')
            sys.exit(0)
        
        else:
            mulseq = []
            i=0
            while i <len(rows) and i%2==0:
                seq = rows[i+1].strip()
                AA_label = [AA_single[i] for i in seq]
                AA_label = np.array(AA_label)
                mulseq.append(AA_label)
                i+=2
            return mulseq




def load_Data(feature_path,inputfile, pdbID, chainID,seq_tobe_designed=None):

    if seq_tobe_designed:
        f = open(seq_tobe_designed)
        rows = f.readlines()
        f.close()
        if rows[0][0] !='>':
            print('Wrong FASTA format for input to be designed sequence')
            sys.exit(0)
        else:
            seq = rows[1].strip()
            AA_label = [AA_single[i] for i in seq]
            AA_label = np.array(AA_label)

    if os.path.exists(  os.path.join(feature_path,  pdbID + '_' +chainID + '_edgeindex')):
        edgeattr = torch.from_numpy( np.loadtxt( os.path.join( feature_path, pdbID + '_' +chainID+'_edgeattr') ) )
        nodefeature = torch.from_numpy( np.loadtxt( os.path.join( feature_path, pdbID + '_' +chainID + '_nodefeature')  ))
        nodectegory = torch.from_numpy( np.loadtxt( os.path.join( feature_path, pdbID + '_' +chainID + '_nodecategory') ) ).long()
        edgeindex = torch.from_numpy( np.loadtxt( os.path.join( feature_path, pdbID + '_' +chainID + '_edgeindex')).T ).long()
        #distance = f2dgenerate( feature_path, inputfile,pdbID,chainID) 
        distance = torch.from_numpy(np.load(os.path.join( feature_path, pdbID + '_' +chainID + '_dismap.npy'  )) )
        #orientation = dihedral_angle(feature_path, pdbID, os.path.join(feature_path, pdbID +'_' + chainID +'.pdb'),  chainID)
        orientation = torch.from_numpy( np.load(os.path.join( feature_path, pdbID + '_' +chainID + '_orientation.npy')) )
    else:
        geometric_f, orientation, distance = pre_fea(feature_path,inputfile,pdbID,chainID)
        [nodefeature, nodectegory, edgeattr, edgeindex] = [i for i in geometric_f]

    if seq_tobe_designed:
        nodectegory = torch.from_numpy( AA_label ).long()


    y_cato = torch.scatter( torch.zeros( nodectegory.size(0),20), 1, nodectegory.unsqueeze(1),1 ).float() 
    physicochemistry = pd.read_csv('../physicochemisty',sep='\s+',header=None)
    PSP19 = np.loadtxt('../PSP19')
    y_pc = torch.from_numpy(physicochemistry.iloc[:,1:].values)[nodectegory]
    y_psp = torch.from_numpy(PSP19)[nodectegory]

    data = Data( x = torch.cat( ( nodefeature, y_pc, y_psp, y_cato ), 1), edge_attr=edgeattr, edge_index=edgeindex,)
    #print(distance.size(),orientation.size())
    data.distance = torch.cat( (distance.unsqueeze(2), orientation), -1)
    
    AA_label = nodectegory.numpy()
    return AA_label, data




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pre_fea(inputPATH,inputfile,pdbname,chainID):
    pool = ThreadPoolExecutor(max_workers=3)
    t1 = pool.submit(lambda p:preprocess_singlechain(*p),[inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID])
    t2 = pool.submit(lambda p:dihedral_angle(*p),[inputPATH,pdbname,os.path.join(inputPATH,inputfile),chainID])
    t3 = pool.submit(lambda p:f2dgenerate(*p),[inputPATH,inputfile,pdbname,chainID])
    t1_r = t1.result()
    t2_r = t2.result()
    t3_r = t3.result()
    return t1_r,t2_r,t3_r





