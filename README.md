# GeoPacker
GeoPacker is a tool package using a simplified geometric deep learning coupling ResNet for protein side-chain modeling. Particularly, GeoPacker is suitable for large scale protein sequence design with side chain modeling due to the admirable efficiency and accuracy.

![Alt text](https://github.com/PKUliujl/GeoPacker/image/4v17A.png)

The dependent packages for GeoPacker are listed in the `requirement.txt`.

INSTALLATION
======================
1. Download it by `git clone https://github.com/PKUliujl/GeoPacker.git` 
2. Unzip the four files of pre-trained model parameters in the `model` folder

Before using GeoPacker, please check whether the dependent packages are available in your environment (see `requirement.txt`)


USAGE
======================
```
usage: run_GeoPacker.py [-h] [--purpose {0,1}] [--inputPATH INPUTPATH] --inputfile INPUTFILE --pdbname PDBNAME  
                        --chainID CHAINID [--seqfile SEQFILE] [--outputfile OUTPUTFILE] [--outputPATH OUTPUTPATH]  

To better use the tool for protein side-chain modeling, please add some of these parameters  

optional arguments:  
    -h, --help            show this help message and exit  
    --purpose {0,1}       0 for repacker while 1 for sequence design    
    --inputPATH INPUTPATH, -iP INPUTPATH    
                          the directory path containing the pdb file  
    --inputfile INPUTFILE, -i INPUTFILE  
                          a pdb file, eg. 1a12.pdb/1a12_A.pdb  
    --pdbname PDBNAME     a protein name, eg. 1A12/1a12  
    --chainID CHAINID     a protein chain to be packered, eg. A  
    --seqfile SEQFILE     a fasta format file including the sequences to be designed  
    --outputfile OUTPUTFILE, -o OUTPUTFILE  
                          the name of output file provided only for repacking. default:  
                          pdbname_chainID_repackered.pdb  
    --outputPATH OUTPUTPATH, -oP OUTPUTPATH  
                          the directory path of the outputfile. default: inputPATH  
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


CITATION
=====================
If you find GeoPacker useful in your research, please cite it:



