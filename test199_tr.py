
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from sklearn import preprocessing
from hat.models import Sequential, Model
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop, Adam
import hat.objectives as obj
import hat.backend as K
import config as cfg
import prepare_data as pp_data
import theano.tensor as T
import cb_eer

# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len
act = 'relu'
n_hid = 500
fold = 4
n_out = len( cfg.labels )

def loss_func( out_nodes, any_nodes, gt_nodes ):
    a8_node = out_nodes[0]      # shape: (n_songs, n_chunk, n_out)
    gt_node = gt_nodes[0]       # shape: (n_songs, n_out)
    
    (n_songs, n_chunk, n_out) = a8_node.shape
    a8_node_2d = a8_node.reshape( (n_songs*n_chunk, n_out) )
    gt_node_2d = T.tile( gt_node, n_chunk ).reshape( (n_songs*n_chunk, n_out) )
    loss_node = obj.get( 'binary_crossentropy' )( a8_node_2d, gt_node_2d )
    return loss_node


def train():
    #scaler = pp_data.GetScaler( fe_fd, fold )
    #tr_X, tr_y, te_X, te_y = pp_data.GetScalerSegData( fe_fd, agg_num, hop, fold, scaler )
    tr_X, tr_y, tr_nas, _, _, _ = pp_data.GetSegData( fe_fd, agg_num, hop, fold )

    
    # prepare data
    #tr_X, tr_y, te_X, te_y = pp_data.GetSegData( fe_fd, agg_num, hop, fold )
    [n_songs, n_chunks, _, n_in] = tr_X.shape
    
    print tr_X.shape, tr_y.shape
    
    '''
    # A.
    in0 = InputLayer( (n_chunks, agg_num, n_in), name='in0' )
    a1 = Flatten( 3, name='a1' )( in0 )
    a2 = Dense( n_hid, act='relu' )( a1 )
    a4 = Dense( n_hid, act='relu' )( a2 )
    a6 = Dense( n_hid, act='relu' )( a4 )
    a8 = Dense( n_out, act='sigmoid', name='a8' )( a6 )
    '''
    
    
    # B.
    in0 = InputLayer( (n_chunks, agg_num, n_in), name='in0' )   # shape: (n_songs, n_chunk, agg_num, n_in)
    a1 = Flatten( 3, name='a1' )( in0 )                 # shape: (n_songs, n_chunk, agg_num*n_in)
    a2 = Dense( n_hid, act='relu' )( a1 )               # shape: (n_songs, n_chunk, n_hid)
    a3 = Dropout( 0.2 )( a2 )
    a4 = Dense( n_hid, act='relu' )( a3 )
    a5 = Dropout( 0.2 )( a4 )
    a6 = Dense( n_hid, act='relu' )( a5 )
    a7 = Dropout( 0.2 )( a6 )
    a8 = Dense( n_out, act='sigmoid', b_init=-1, name='a8' )( a7 ) # shape: (n_songs, n_chunk, n_out)
    
    
    md = Model( in_layers=[in0], out_layers=[a8], any_layers=[] )
    
    # callbacks
    # tr_err, te_err are frame based. To get event based err, run recognize.py
    dump_fd = cfg.scrap_fd + '/Results_tr/bagofblock_eer/fold'+str(fold)
    print_scores = cb_eer.PrintScoresBagOfBlocks( tr_X, tr_nas, dump_fd, call_freq=1 )
    save_model = SaveModel( dump_fd=cfg.scrap_fd+'/Md', call_freq=10 )
    callbacks = [ save_model, print_scores ]
    
    # optimizer
    # optimizer = SGD( 0.01, 0.95 )
    optimizer = Adam(2e-4)
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=10, n_epochs=301, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks )
    
    
if __name__ == '__main__':
    train()