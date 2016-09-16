'''
SUMMARY:  the joint detection-classification (JDC) model on development set using cross validation
AUTHOR:   Qiuqiang Kong
Created:  2016.09.12
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from sklearn import preprocessing
from hat.models import Sequential, Model
from hat.layers.core import InputLayer, Flatten, Dense, Dropout, Lambda
from hat.layers.rnn import *
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop, Adam
import hat.objectives as obj
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data
import theano.tensor as T
import cb_eer

# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len
act = 'relu'
n_hid = 500
fold = 1
n_out = len( cfg.labels )
    
# create empty folders in workspace
def create_folders():
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Md_dev_jdc' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev/jdc_eer' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev/jdc_eer/fold' + str(fold) )
    

# loss function
def loss_func( out_nodes, any_nodes, gt_nodes ):
    eps = 1e-6
    a8_node = out_nodes[0]      # shape: (n_songs, n_chunk, n_out)
    b8_node = out_nodes[1]      # shape: (n_songs, n_chunk, n_out)
    gt_node = gt_nodes[0]       # shape: (n_songs, n_out)

    b8_node = T.clip( b8_node, eps, 1-eps )     # clip to avoid numerical underflow
    weighted_out = K.sum( a8_node*b8_node, axis=1 ) / K.sum( b8_node, axis=1 )
    loss_node = obj.get('binary_crossentropy')( weighted_out, gt_node )
    return loss_node


# mean pool
def mean_pool( input ):
    return K.mean( input, axis=2 )
    
    
# train & validate model
def train():
    
    # create empty folders in workspace
    create_folders()
    
    # prepare data
    tr_X, tr_y, _, va_X, va_y, va_na_list = pp_dev_data.GetSegData( fe_fd, agg_num, hop, fold )
    [n_songs, n_chunks, _, n_in] = tr_X.shape
    
    print tr_X.shape, tr_y.shape
    print va_X.shape, va_y.shape
    
    
    # model
    # classifier
    in0 = InputLayer( (n_chunks, agg_num, n_in), name='in0' )   # shape: (n_songs, n_chunk, agg_num, n_in)
    a1 = Flatten( 3, name='a1' )( in0 )         # shape: (n_songs, n_chunk, agg_num*n_in)
    a2 = Dense( n_hid, act='relu' )( a1 )       # shape: (n_songs, n_chunk, n_hid)
    a3 = Dropout( 0.2 )( a2 )
    a4 = Dense( n_hid, act='relu' )( a3 )
    a5 = Dropout( 0.2 )( a4 )
    a6 = Dense( n_hid, act='relu' )( a5 )
    a7 = Dropout( 0.2 )( a6 )
    a8 = Dense( n_out, act='sigmoid', b_init=-1, name='a8' )( a7 )     # shape: (n_songs, n_chunk, n_out)

    # detector
    b1 = Lambda( mean_pool )( in0 )     # shape: (n_songs, n_chunk, n_out)
    b8 = Dense( n_out, act='sigmoid', name='b4' )( b1 )     # shape: (n_songs, n_chunk, n_out)
    
    md = Model( in_layers=[in0], out_layers=[a8, b8], any_layers=[] )
    md.summary()
    
    # callback, write out dection scores to .txt each epoch
    dump_fd = cfg.scrap_fd + '/Results_dev/jdc_eer/fold'+str(fold)
    print_scores = cb_eer.PrintScoresDetectionClassification( va_X, va_na_list, dump_fd, call_freq=1 )
    
    # callback, print loss each epoch
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=va_X, va_y=va_y, te_x=None, te_y=None, 
                            metrics=[loss_func], call_freq=1, dump_path=None )
                            
    # callback, save model every N epochs
    save_model = SaveModel( dump_fd=cfg.scrap_fd+'/Md_dev_jdc', call_freq=10 )
    
    # combine all callbacks
    callbacks = [ validation, save_model, print_scores ]
    
    # optimizer
    # optimizer = SGD( 0.01, 0.95 )
    optimizer = Adam(2e-4)
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=10, n_epochs=301, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks )
    
    
if __name__ == '__main__':
    train()