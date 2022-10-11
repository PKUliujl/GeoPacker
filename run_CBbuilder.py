#!/lustre3/lhlai_pkuhpc/liujl/py38env/bin/python
"""
Created on The Apr 28 19:34:05 2022
 
@author: liujl
"""

import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
GeoPacker_filepath = os.path.abspath(__file__) 
sys.path.append(os.path.join(os.path.dirname(GeoPacker_filepath),'builder/') )
from builder.CB_Rebuilder import *
from sys import argv

if len(argv) !=4:
    print('\n----------\nThis script is used for rebuilding pseudo CB atoms given protein backbone atoms (i.e. C,N,O,CA).\n\nusage: python %s Inputfile(pdb format) chainID Outputfile(pdb format) \n\n----------\n'%argv[0])
    sys.exit()

CB_REbuilder( argv[1], argv[2],  argv[3])
