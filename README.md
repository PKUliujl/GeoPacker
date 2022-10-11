# GeoPacker
GeoPacker is a tool package using a simplified geometric deep learning coupled ResNet for protein side-chain modeling. Particularly, GeoPacker overcomes the limitation of large-scale rotamer-free protein sequence design due to the admirable efficiency and accuracy.

![Alt text](https://github.com/PKUliujl/GeoPacker/blob/main/image/3mpcA.png)

The dependent packages for GeoPacker are listed in the [requirement.txt](https://github.com/PKUliujl/GeoPacker/blob/main/requirement.txt).


INSTALLATION
======================
1. Users can download it by `git clone https://github.com/PKUliujl/GeoPacker.git` (without pretrained model's parameters in the [model](https://github.com/PKUliujl/GeoPacker/blob/main/model) directory  due to the large size), alternatively, 
it is available to access [our web](http://mdl.ipc.pku.edu.cn/) to get the full toolkit;
2. Modify the PATH of python interpreter in the first line in `run_GeoPacker.py`;
3. Add the directory PATH including `run_GeoPacker.py` to your environment PATH, and then type/run `run_GeoPacker.py` in any directory for friendly usage.

Before using GeoPacker, please check whether the dependent packages are available in your environment (see [requirement.txt](https://github.com/PKUliujl/GeoPacker/blob/main/requirement.txt)). If not, using `pip` or `conda` to install them.


USAGE
======================
```
usage: run_GeoPacker.py [-h] [--purpose {0,1}] [--model_identity {pdb30,pdb60}] [--inputPATH INPUTPATH] --inputfile INPUTFILE --pdbname PDBNAME  
                        --chainID CHAINID [--seqfile SEQFILE] [--outputfile OUTPUTFILE] [--outputPATH OUTPUTPATH]  

To better use GeoPacker for protein side-chain modeling, please add some of these parameters  

optional arguments:  
    -h, --help            show this help message and exit  
    --purpose {0,1}       0 for repacking while 1 for sequence design. default: 0
    --model_identity {pdb30,pdb60}
                          model trained on different pair-wise sequence identity dataset. default: pdb60
    --inputPATH INPUTPATH, -iP INPUTPATH    
                          the directory path containing the pdb file. default: './'  
    --inputfile INPUTFILE, -i INPUTFILE  
                          a pdb file under inputPATH, eg. 1a12.pdb/1a12_A.pdb  
    --pdbname PDBNAME     a protein name, eg. 1A12/1a12  
    --chainID CHAINID     a protein chain to be packed, eg. A  
    --seqfile SEQFILE     a fasta format file including the sequences to be designed  
    --outputPATH OUTPUTPATH, -oP OUTPUTPATH  
                          the directory path of the outputfile. default: inputPATH
    --outputfile OUTPUTFILE, -o OUTPUTFILE  
                          the name of output file. default: pdbname_chainID_repacked.pdb for repacking 
                          and pdbname_chainID_design.pdb for design
  
```

EXAMPLE
=====================
For repacking, 
```python
      python run_GeoPacker.py --inputPATH example/ -i 3MPC_A.pdb --pdbname 3MPC  --chainID A   
```

For design,  
```python
      python run_GeoPacker.py --purpose 1 --inputPATH example/ -i 3MPC_A.pdb --pdbname 3MPC --chainID A --seqfile  example/seqfile
```

Attention, do not use following commands for design, otherwise, error will be reported:
```python
      python run_GeoPacker.py --purpose 1 --inputPATH example/ -i example/3MPC_A.pdb --pdbname 3MPC --chainID A --seqfile  example/seqfile
      python run_GeoPacker.py --purpose 1 --inputPATH example/ -i 3MPC_A.pdb --pdbname 3MPC --chainID A --seqfile  seqfile
      python run_GeoPacker.py --purpose 1  -i example/3MPC_A.pdb --pdbname 3MPC --chainID A --seqfile  example/seqfile
```

To rebuild pseudo CB atoms given protein backbone atoms (i.e. C, N, O, CA), 
```
      run_CBbuilder.py Inputfile(pdb format) chainID Outputfile(pdb format)
      example: run_CBbuilder.py example/3MPC_A.pdb A example/3MPC_A.CB.pdb
```

Reminder: Only the regular pdb format files are accepted as inputs. 

Feel free to contact me via email at liujl@stu.pku.edu.cn for other issues.  

ACKNOWLEDGEMENT
=====================
A part of the code used to reconstruct the side chain atomic coordinates was taken from [PeptideBuilder](https://peerj.com/articles/80/) and [opus-rota4](https://academic.oup.com/bib/article/23/1/bbab529/6461160?searchresult=1) and modified to meet our requirements.

CITATION
=====================
If you find GeoPacker useful in your research, please cite it:



