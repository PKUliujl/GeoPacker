# GeoPacker
GeoPacker is a tool package using a simplified geometric deep learning coupled ResNet for protein side-chain modeling. Particularly, GeoPacker is suitable for large scale protein side chain modeling of backbone-fixed generated sequences due to the admirable efficiency and accuracy.

![Alt text](https://github.com/PKUliujl/GeoPacker/blob/main/image/3mpcA.png)

The dependent packages for GeoPacker are listed in the `requirement.txt`.

INSTALLATION
======================
1. Download it by `git clone https://github.com/PKUliujl/GeoPacker.git` 
2. Unzip the four files of pre-trained model parameters in the `model` folder

Before using GeoPacker, please check whether the dependent packages are available in your environment (see `requirement.txt`). If not, using `pip` or `conda` to install them.


USAGE
======================
```
usage: run_GeoPacker.py [-h] [--purpose {0,1}] [--inputPATH INPUTPATH] --inputfile INPUTFILE --pdbname PDBNAME  
                        --chainID CHAINID [--seqfile SEQFILE] [--outputfile OUTPUTFILE] [--outputPATH OUTPUTPATH]  

To better use the tool for protein side-chain modeling, please add some of these parameters  

optional arguments:  
    -h, --help            show this help message and exit  
    --purpose {0,1}       0 for repacker while 1 for sequence design. default: 0    
    --inputPATH INPUTPATH, -iP INPUTPATH    
                          the directory path containing the pdb file. default: './'  
    --inputfile INPUTFILE, -i INPUTFILE  
                          a pdb file under inputPATH, eg. 1a12.pdb/1a12_A.pdb  
    --pdbname PDBNAME     a protein name, eg. 1A12/1a12  
    --chainID CHAINID     a protein chain to be packered, eg. A  
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

Reminder: Only the regular pdb format files are accepted as inputs. If an error occurs, please delete the intermediate output files and then rewrite the new parameters. Feel free to contact me via email at liujl@stu.pku.edu.cn for other issues.  

CITATION
=====================
If you find GeoPacker useful in your research, please cite it:



