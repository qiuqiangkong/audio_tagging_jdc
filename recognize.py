'''
SUMMARY:  Dcase 2016 Task 4. Audio Tagging
          Recognize and evaluate the f value or EER
AUTHOR:   Qiuqiang Kong
Created:  2016.05.29
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from hat.optimizers import Rmsprop
from hat.metrics import prec_recall_fvalue
from hat import serializations
import hat.backend as K
import config as cfg
import prepare_data as pp_data
import csv
import cPickle
from main_dnn import fe_fd, agg_num, hop, n_hid, fold

# hyper-params
n_labels = len( cfg.labels )

# load model
md = serializations.load( cfg.scrap_fd + '/Md/md0.p' )

def recognize():
    # prepare data
    _, _, te_X, te_y = pp_data.GetAllData( fe_fd, agg_num, hop, fold )
    
    # do recognize and evaluation
    thres = 0.4     # thres, tune to prec=recall
    n_labels = len( cfg.labels )
    
    gt_roll = []
    pred_roll = []
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            if fold==curr_fold:
                # get features, tags
                fe_path = fe_fd + '/' + na + '.f'
                info_path = cfg.dev_wav_fd + '/' + na + '.csv'
                tags = pp_data.GetTags( info_path )
                y = pp_data.TagsToCategory( tags )
                X = cPickle.load( open( fe_path, 'rb' ) )
                
                # aggregate data
                X3d = mat_2d_to_3d( X, agg_num, hop )
        
                p_y_pred = md.predict( X3d )
                p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
                pred = np.zeros(n_labels)
                pred[ np.where(p_y_pred>thres) ] = 1
                pred_roll.append( pred )
                gt_roll.append( y )
    
    pred_roll = np.array( pred_roll )
    gt_roll = np.array( gt_roll )
    
    # calculate prec, recall, fvalue
    prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
    print prec, recall, fvalue

if __name__ == '__main__':
    recognize()
