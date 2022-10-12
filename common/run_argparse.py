
import argparse
import os

def run_inputparameters():
    description='To better use GeoPacker for protein side-chain modeling, please add some of these parameters'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--purpose',type=int,choices=[0,1],help='0 for repacking while 1 for sequence design. default: 0',default=0)
    parser.add_argument('--model_identity',type=str,choices=['pdb30','pdb60'],help='model trained on different pair-wise sequence identity dataset. default: pdb60',default='pdb60')
    parser.add_argument('--inputPATH','-iP',type=str,help='the directory path containing the pdb file. default: "./"',default = './')
    parser.add_argument('--inputfile','-i',type=str,help='a pdb file under inputPATH, eg. 1a12.pdb/1a12_A.pdb',required=True)
    parser.add_argument('--pdbname',type=str,required=True, help='a protein name, eg. 1A12/1a12')
    parser.add_argument('--chainID',type=str,required=True, help='a protein chain to be packed, eg. A')
    parser.add_argument('--seqfile',type=str,help='a fasta format file including the sequence to be designed')
    parser.add_argument('--outputfile','-o',type=str,help='the name of output file. default: pdbname_chainID_repacked.pdb for repacking and pdbname_chainID_design.pdb for design')
    parser.add_argument('--outputPATH','-oP',type=str,help='the directory path of the outputfile. default: inputPATH')
    args = parser.parse_args()
    
    if args.purpose == 1:
        if args.seqfile is None :
            parser.error('With design purpose, a seqfile is required')
    
    if args.seqfile:
        args.purpose = 1
    
    if args.outputPATH is None:
        args.outputPATH = args.inputPATH

    if args.outputfile is None and args.purpose==0:
        args.outputfile =  args.pdbname +'_' + args.chainID + '_repacked.pdb'
    
    if args.outputfile is None and args.purpose==1:
        args.outputfile =  args.pdbname +'_' + args.chainID + '_design.pdb'
        

    return  args

    
if __name__=='__main__':
    run_inputparameters()
