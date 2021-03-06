'''
SUMMARY:  the bag of blocks (BOB) model on development set using cross validation
AUTHOR:   Qiuqiang Kong
Created:  2016.09.12
Modified: 2016.10.09 Update
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from sklearn import preprocessing
from hat.models import Sequential, Model
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import Adam
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
fold = 0
n_out = len( cfg.labels )


# create empty folders in workspace
def create_folders():
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Md_dev_bob' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev/bob_eer' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_dev/bob_eer/fold' + str(fold) )


# the BOB model loss function
def loss_func( md ):
    a8_node = md.out_nodes_[0]      # shape: (n_songs, n_chunk, n_out)
    gt_node = md.gt_nodes_[0]       # shape: (n_songs, n_out)
    
    (n_songs, n_chunk, n_out) = a8_node.shape
    a8_node_2d = a8_node.reshape( (n_songs*n_chunk, n_out) )
    gt_node_2d = T.tile( gt_node, n_chunk ).reshape( (n_songs*n_chunk, n_out) )
    loss_node = obj.get( 'binary_crossentropy' )( a8_node_2d, gt_node_2d )
    return loss_node


def train():
    
    # create empty folders in workspace
    create_folders()
    
    # get train & valid data on the k-th fold
    tr_X, tr_y, _, va_X, va_y, va_na_list = pp_dev_data.GetSegData( fe_fd, agg_num, hop, fold )

    [n_songs, n_chunks, _, n_in] = tr_X.shape
    print tr_X.shape, tr_y.shape
    print va_X.shape, va_y.shape
    
    # model
    lay_in0 = InputLayer( (n_chunks, agg_num, n_in), name='in0' )   # shape: (n_songs, n_chunk, agg_num, n_in)
    lay_a1 = Flatten( 3, name='a1' )( lay_in0 )                 # shape: (n_songs, n_chunk, agg_num*n_in)
    lay_a2 = Dense( n_hid, act='relu' )( lay_a1 )               # shape: (n_songs, n_chunk, n_hid)
    lay_a3 = Dropout( 0.2 )( lay_a2 )
    lay_a4 = Dense( n_hid, act='relu' )( lay_a3 )
    lay_a5 = Dropout( 0.2 )( lay_a4 )
    lay_a6 = Dense( n_hid, act='relu' )( lay_a5 )
    lay_a7 = Dropout( 0.2 )( lay_a6 )
    lay_a8 = Dense( n_out, act='sigmoid', b_init=-1, name='a8' )( lay_a7 ) # shape: (n_songs, n_chunk, n_out)
    
    md = Model( in_layers=[lay_in0], out_layers=[lay_a8], any_layers=[] )
    md.compile()
    md.summary()
    
    
    # callback, write out dection scores to .txt each epoch
    dump_fd = cfg.scrap_fd + '/Results_dev/bob_eer/fold'+str(fold)
    print_scores = cb_eer.PrintScoresBagOfBlocks( va_X, va_na_list, dump_fd, call_freq=1 )
    
    # callback, print loss each epoch
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=va_X, va_y=va_y, te_x=None, te_y=None, 
                            metrics=[loss_func], call_freq=1, dump_path=None )
                            
    # callback, save model every N epochs
    save_model = SaveModel( dump_fd=cfg.scrap_fd+'/Md_dev_bob', call_freq=10 )
    
    # combine all callbacks
    callbacks = [ validation, save_model, print_scores ]
    
    # optimizer
    optimizer = Adam(2e-4)
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=10, n_epochs=301, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks )
    
    
if __name__ == '__main__':
    train()