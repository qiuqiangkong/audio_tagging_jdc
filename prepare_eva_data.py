'''
SUMMARY:  prepare data for evaluation
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/hat')
import numpy as np
import csv
import cPickle
import pickle
import prepare_dev_data as pp_dev_data
import config as cfg
from hat.preprocessing import mat_2d_to_3d
import os
import eer

def GetAllData( fe_fd, csv_file, agg_num, hop ):
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        Xlist.append( X3d )

    return np.concatenate( Xlist, axis=0 )
    
# size: n_songs*n_chunks*agg_num*n_in
def GetEvaSegData( fe_fd, agg_num, hop ):
    te_Xlist = []
        
    names = os.listdir( fe_fd )
    te_na_list = []
        
    # read one line
    for na in names:
        fe_path = fe_fd + '/' + na
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        te_Xlist.append( X3d )
        te_na_list.append( na[0:-2] )

    return np.array( te_Xlist ), te_na_list


### calculate eer from score file
def calculate_eva_eer( n_epochs, preds_fd, dump_path ):
    csv_path = cfg.eva_csv_path
    anno_fd = cfg.eva_anno_fd

    mean_eer_list = []
    for epoch in xrange( n_epochs ):
        print epoch,
        pred_path = preds_fd + '/' + str(epoch) + 'epoch.txt'
        
        # get ground truth dict
        gt_dict = {}
        with open( csv_path, 'rb') as f:
            reader = csv.reader(f)
            lis = list(reader)
            
            for li in lis:
                na = li[1]
                    
                anno_path = anno_fd + '/' + na + '.csv'
                tags = pp_dev_data.GetTags( anno_path )
                gt_dict[na+'.16kHz.wav'] = tags
        
        # evaluate eer for each tag and average them
        eer_ary = []
        for tag in cfg.labels:
            if tag is not 'S':
                gt_dict_curr = {}
                for key in gt_dict.keys():
                    if tag in gt_dict[key]:
                        gt_dict_curr[key] = 1
                    else:
                        gt_dict_curr[key] = 0
                
                eer_val = eer.compute_eer( pred_path, tag, gt_dict_curr )
                eer_ary.append( eer_val )
        
        # average err of each tag
        mean_eer_list.append( np.mean(eer_ary) )
    
    # dump err list of all epochs
    pickle.dump( mean_eer_list, open( dump_path, 'wb' ) )


if __name__ == "__main__":
    pp_dev_data.CreateFolder( cfg.eva_fe_fd )
    pp_dev_data.CreateFolder( cfg.eva_fe_mel_fd )
    pp_dev_data.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=0 )
    