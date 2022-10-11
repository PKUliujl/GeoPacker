
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import torch
from MainScripts import *
import sys
sys.path.append('MainScripts')
sys.path.append('builder/')
from MainScripts.utils import *
from common.run_argparse import *
from distributionR.samplingR import samplingr
from builder.test import builder
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = torch.load('model/157_0_ft16.h5', map_location = device)
model2 = torch.load('model/157_1_ft16.h5', map_location = device)
model3 = torch.load('model/157_2_ft16.h5', map_location = device)
model4 = torch.load('model/157_3_ft16.h5', map_location = device)



def evaluate( feature_path, pdbname, chainID,seq_tobe_designed=None,seqname=None,packered_outPATH=None,outputfile=None ):
    print('\nInput: %s    %s   purpose: %d'%(os.path.join(feature_path,inputfile), pdbname+chainID, args.purpose) )
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
    AA_label, data = load_Data(feature_path, inputfile,pdbname, chainID ,seq_tobe_designed)
    print('Side chain dihedral angle prediction...', )
    with torch.no_grad():

        data = data.to(device)
        AA =  AA_label       
        pred1,dis1,orien1 = model1(x = data.x.float(), edge_index = data.edge_index, edge_attr = data.edge_attr.float(),distance = data.distance.float(), ) #distance = data.distance.float()
        pred2,dis2,orien2 = model2(x = data.x.float(), edge_index = data.edge_index, edge_attr = data.edge_attr.float(),distance = data.distance.float(), )
        pred3,dis3,orien3 = model3(x = data.x.float(), edge_index = data.edge_index, edge_attr = data.edge_attr.float(),distance = data.distance.float(), )
        pred4,dis4,orien4 = model4(x = data.x.float(), edge_index = data.edge_index, edge_attr = data.edge_attr.float(),distance = data.distance.float(), )
        pred = ( torch.nn.functional.softmax(pred1,1) + torch.nn.functional.softmax(pred2,1) + torch.nn.functional.softmax(pred3,1) + torch.nn.functional.softmax(pred4,1))/4 # + torch.nn.functional.softmax(pred5,1) + torch.nn.functional.softmax(pred6,1) + torch.nn.functional.softmax(pred7,1) + torch.nn.functional.softmax(pred8,1) + torch.nn.functional.softmax(pred9,1)  )/9
        #np.save('dis',torch.nn.Softmax(1)(dis1).max(1)[1].cpu().squeeze(0).numpy())
        #np.save('orientation',torch.nn.Softmax(1)(orien1).max(1)[1].cpu().squeeze(0).numpy())
        pred = pred[:,:48,:].argmax(1)
        rotamers = []
        j = 0
        while j < len(data.x):
            for rn in range(  Rotamer_number[ AA[j] ]) :
                sampling_interval = pred[j,rn].item()
                res = standard_aa_names_r[ AA[j] ]
                angle = samplingr( res, rn, sampling_interval)
                rotamers.append( float(angle) )
            j+=1
            
        print( 'Writing to pdb file ( %s ) ...'%os.path.join(packered_outPATH,outputfile) )
        if seq_tobe_designed and packered_outPATH:
            builder( os.path.join( feature_path, inputfile),chainID, rotamers, os.path.join( packered_outPATH, outputfile),seq_tobe_designed)
        else:
            builder( os.path.join( feature_path, inputfile), chainID,rotamers, os.path.join( packered_outPATH, outputfile) )
        print('Done!',)




if  __name__ =='__main__':
    args = run_inputparameters()
    feature_path = args.inputPATH
    chainID = args.chainID
    pdbname = args.pdbname
    inputfile = args.inputfile
    seq_tobe_designed = args.seqfile
    packered_outPATH = args.outputPATH
    outputfile = args.outputfile
    if args.purpose == 1:
        #seqs = read_seqfile(seq_tobe_designed)
        #for i in range(len(seqs)):
        evaluate(feature_path, pdbname, chainID,seq_tobe_designed, pdbname+'_'+chainID+'_'+'design',packered_outPATH, outputfile)
    else:
        seqname=None
        evaluate(feature_path, pdbname, chainID,seq_tobe_designed,seqname,packered_outPATH,outputfile)
    print(' '*13+'##############################################\n\
             ##      Thanks for using GeoPacker          ##\n\
             ##    More details see mdl.ipc.pku.edu.cn   ##\n\
             ##############################################\n')
