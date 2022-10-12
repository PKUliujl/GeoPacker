

import numpy as np
import os,Bio
import Bio.PDB
from Bio.PDB.vectors import calc_dihedral
from math import *
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from multiprocessing import Process,Queue
from torch_geometric.data import Data
import torch

import warnings
warnings.filterwarnings('ignore')

from single_f import *
from complex_f import *


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
    #np.save( os.path.join(path, pdbname + '_' + chainID+'_dismap') ,a)
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


path = os.path.split(os.path.realpath(__file__)) [0]

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

    if os.path.exists(os.path.join(os.path.abspath(feature_path),'tmp_features/%s'%(pdbID + '_' +chainID + '_features.npz'))):
        print('Precomputed geometric features from protein backbone atoms existed in the file %s and it will be loaded directly for side-chain conformation prediction'%(os.path.join(os.path.abspath(feature_path),'tmp_features/%s'%(pdbID + '_' +chainID + '_features.npz'))))
        
        featuresfile = np.load(os.path.join(os.path.abspath(feature_path),'tmp_features/%s'%(pdbID + '_' +chainID + '_features.npz')))
        edgeattr = torch.from_numpy( featuresfile['edgeattr'] )
        nodefeature = torch.from_numpy( featuresfile['nodefeature'] )
        nodectegory = torch.from_numpy( featuresfile['nodectegory'] )
        edgeindex = torch.from_numpy( featuresfile['edgeindex'] )
        distance = torch.from_numpy( featuresfile['distance'] )
    else:
        if not os.path.exists(os.path.join(os.path.abspath(feature_path),'tmp_features')):
            os.mkdir(os.path.join(os.path.abspath(feature_path),'tmp_features'))
        
        geometric_f, distance = pre_fea(feature_path,inputfile,pdbID,chainID)
        [nodefeature, nodectegory, edgeattr, edgeindex] = [i for i in geometric_f]
        np.savez( os.path.join(os.path.abspath(feature_path),'tmp_features/%s'%(pdbID + '_' +chainID + '_features.npz')), edgeattr = edgeattr.numpy(),nodefeature = nodefeature.numpy(),nodectegory = nodectegory.numpy(), edgeindex = edgeindex.numpy(), distance = distance.numpy())
        print('Creat a directory "tmp_features" in %s and a temporary file %s will be saved in this directory'%(os.path.abspath(feature_path),pdbID + '_' +chainID + '_features.npz'))
                
                

    if seq_tobe_designed:
        print('Sequence to be designed: ',os.popen(f'cat {seq_tobe_designed}').readlines()[1].strip())
        nodectegory = torch.from_numpy( AA_label ).long()


    y_cato = torch.scatter( torch.zeros( nodectegory.size(0),20), 1, nodectegory.unsqueeze(1),1 ).float() 
    physicochemistry = pd.read_csv(os.path.join(path,'../common/physicochemisty'),sep='\s+',header=None)
    PSP19 = np.loadtxt(os.path.join(path,'../common/PSP19'))
    y_pc = torch.from_numpy(physicochemistry.iloc[:,1:].values)[nodectegory]
    y_psp = torch.from_numpy(PSP19)[nodectegory]

    data = Data( x = torch.cat( ( nodefeature, y_pc, y_psp, y_cato ), 1), edge_attr=edgeattr, edge_index=edgeindex,)
    #print(distance.size(),orientation.size())
    data.distance = distance.unsqueeze(2)
    
    AA_label = nodectegory.numpy()
    return AA_label, data




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pre_fea(inputPATH,inputfile,pdbname,chainID):
    dis = f2dgenerate(inputPATH,inputfile,pdbname,chainID)
    lengths = dis.size(0)
    intervals = np.linspace(0,lengths,8)
    #intervals2 = np.linspace(0,lengths,16)

    pool = ProcessPoolExecutor(max_workers=7)
    
    if len(chainID)<=1:
        t1 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[0],intervals[1]  )
        t2 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[1],intervals[2]  )
        t3 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[2],intervals[3]  )
        t4 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[3],intervals[4]  )
        t5 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[4],intervals[5]  )
        t6 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[5],intervals[6]  )
        t7 = pool.submit(preprocess_singlechain,inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[6],intervals[7]  )
    else:
        t1 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[0],intervals[1]  )
        t2 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[1],intervals[2]  )
        t3 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[2],intervals[3]  )
        t4 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[3],intervals[4]  )
        t5 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[4],intervals[5]  )
        t6 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[5],intervals[6]  )
        t7 = pool.submit(preprocess_complex, inputPATH,pdbname, os.path.join(inputPATH,inputfile), chainID, intervals[6],intervals[7]  )


    t1_r = t1.result()
    t2_r = t2.result()
    t3_r = t3.result()
    t4_r = t4.result()
    t5_r = t5.result()
    t6_r = t6.result()
    t7_r = t7.result()
    '''
    t8_r = t8.result()
    t9_r = t9.result()
    t10_r = t10.result()
    t11_r = t11.result()
    t12_r = t12.result()
    t13_r = t13.result()
    t14_r = t14.result()
    t15_r = t15.result()
    t16_r = t16.result()
    t17_r = t17.result()
    t18_r = t18.result()
    '''
    t_r2 = torch.cat((t1_r[0], t2_r[0],t3_r[0], t4_r[0],t5_r[0], t6_r[0],t7_r[0]),axis=0),torch.cat((t1_r[1], t2_r[1],t3_r[1],t4_r[1],t5_r[1],t6_r[1],t7_r[1]),axis=0),torch.cat((t1_r[2], t2_r[2],t3_r[2],t4_r[2],t5_r[2],t6_r[2],t7_r[2]),axis=0), torch.cat((t1_r[3], t2_r[3],t3_r[3],t4_r[3],t5_r[3],t6_r[3],t7_r[3]),axis=1)


    return t_r2,dis

