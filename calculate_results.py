'''
SUMMARY:  Calculate eer results from all epochs' scores
AUTHOR:   Qiuqiang Kong
Created:  2016.09.12
Modified: -
--------------------------------------
'''
import numpy as np
import pickle
import csv
import os
import config as cfg
import prepare_dev_data as pp_dev_data
import prepare_eva_data as pp_eva_data


### calculate eer for dev | eva, bob | jdc

def calculate_dev_bob_eer():
    n_epochs = 10
    
    for fold in xrange( cfg.n_folds ):
        preds_fd = cfg.scrap_fd + '/Results_dev/bob_eer/fold' + str(fold)
        if not os.path.exists( preds_fd ):
            print fold, 'fold does not exist!'
        else:
            dump_path = cfg.scrap_fd + '/Results_dev/bob_eer/eer_fold' + str(fold) + '.p'
            pp_dev_data.calculate_dev_eer( n_epochs, preds_fd, dump_path, fold )
            print fold, 'fold finished. You can pickle.load to view result.'
        
def calculate_dev_jdc_eer():
    n_epochs = 10    
    
    for fold in xrange( cfg.n_folds ):
        preds_fd = cfg.scrap_fd + '/Results_dev/jdc_eer/fold' + str(fold)
        if not os.path.exists( preds_fd ):
            print fold, 'fold does not exist!'
        else:
            dump_path = cfg.scrap_fd + '/Results_dev/jdc_eer/eer_fold' + str(fold) + '.p'
            pp_dev_data.calculate_dev_eer( n_epochs, preds_fd, dump_path, fold )
            print fold, 'fold finished. You can pickle.load to view result.'
        
def calculate_eva_bob_eer():
    n_epochs = 10    
    preds_fd = cfg.scrap_fd + '/Results_eva/bob_eer'
    dump_path = cfg.scrap_fd + '/Results_eva/bob_eer.p'
    
    pp_eva_data.calculate_eva_eer( n_epochs, preds_fd, dump_path )
    
def calculate_eva_jdc_eer():
    n_epochs = 10    
    preds_fd = cfg.scrap_fd + '/Results_eva/jdc_eer'
    dump_path = cfg.scrap_fd + '/Results_eva/jdc_eer.p'
    
    pp_eva_data.calculate_eva_eer( n_epochs, preds_fd, dump_path )
    
    
### main (run either of below)
calculate_dev_bob_eer()
#calculate_dev_jdc_eer()
#calculate_eva_bob_eer()
#calculate_eva_jdc_eer()