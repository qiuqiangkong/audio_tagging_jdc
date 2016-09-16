'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.preprocessing import mat_2d_to_3d
import numpy as np
from scipy import signal
import cPickle
import pickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
import wavio
import librosa
import config as cfg
import csv
import scipy.stats
from sklearn import preprocessing
from hat.metrics import prec_recall_fvalue
import eer

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        
        # to mono
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        
        # get spectrogram
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' )
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        
        # save out feature
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
          
### format label
# read tags from csv file
def GetTags( csv_path ):
    with open( csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]       # majority vote of the tags
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y

###
# get chunk data, size: N*agg_num*n_in
def GetAllData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        csv_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( csv_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ] * len( X3d )
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ] * len( X3d )

    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )
    
# get train & valid data from development set
#   size(X): n_songs*n_chunks*agg_num*n_in
#   size(y): n_songs
def GetSegData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
    tr_na_list, te_na_list = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        csv_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( csv_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        # valid data
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
            te_na_list.append( na )
        # train data
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]
            tr_na_list.append( na )

    return np.array( tr_Xlist ), np.array( tr_ylist ), tr_na_list, \
           np.array( te_Xlist ), np.array( te_ylist ), te_na_list
           
def GetScaler( fe_fd, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        if curr_fold!=fold:
            tr_Xlist.append( X )
            
    Xall = np.concatenate( tr_Xlist, axis=0 )
    scaler = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( Xall )

    return scaler
           
def GetScalerSegData( fe_fd, agg_num, hop, fold, scaler ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        csv_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( csv_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        if scaler is not None:
            X = scaler.transform( X )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]

    return np.array( tr_Xlist ), np.array( tr_ylist ), \
           np.array( te_Xlist ), np.array( te_ylist )


### calculate eer from score file
def calculate_dev_eer( n_epochs, preds_fd, dump_path, fold ):
    csv_path = cfg.dev_cv_csv_path
    anno_fd = cfg.dev_anno_fd

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
                curr_fold = int(li[2])
                    
                if fold == curr_fold:
                    anno_path = anno_fd + '/' + na + '.csv'
                    tags = GetTags( anno_path )
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
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
if __name__ == "__main__":
    CreateFolder( cfg.dev_fe_fd )
    CreateFolder( cfg.dev_fe_mel_fd )
    GetMel( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )
