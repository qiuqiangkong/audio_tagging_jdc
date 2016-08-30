'''
SUMMARY:  Dcase 2016 Task 4. Audio Tagging
          Training time: 1 s/epoch. (Tesla M2090)
          test f_value: 73% (thres=0.5), test EER=24%  after 50 epoches     
          Try adjusting hyper-params, optimizer, longer epoches to get better results. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.29
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from sklearn import preprocessing
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop, Adam
import hat.backend as K
import config as cfg
import prepare_data as pp_data


# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
fold = 1
n_out = len( cfg.labels )

def train():
    # prepare data
    tr_X, tr_y, te_X, te_y = pp_data.GetAllData( fe_fd, agg_num, hop, fold )
    [batch_num, n_time, n_freq] = tr_X.shape
    
    print tr_X.shape, tr_y.shape
    print te_X.shape, te_y.shape
    
    # build model
    seq = Sequential()
    seq.add( InputLayer( (n_time, n_freq) ) )
    seq.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_hid, act=act ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_hid, act=act ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_hid, act=act ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_out, act='sigmoid' ) )
    md = seq.combine()
    md.summary()
    
    # callbacks
    # tr_err, te_err are frame based. To get event based err, run recognize.py
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, 
                            metrics=['binary_crossentropy'], call_freq=1, dump_path=cfg.scrap_fd+'/Results/validation.p' )
    save_model = SaveModel( dump_fd=cfg.scrap_fd+'/Md', call_freq=10 )
    callbacks = [ validation, save_model ]
    
    # optimizer
    # optimizer = SGD( 0.01, 0.95 )
    optimizer = Adam(1e-4)
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=100, n_epochs=101, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )
    
    
if __name__ == '__main__':
    train()
