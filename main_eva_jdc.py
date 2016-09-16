'''
SUMMARY:  the joint detection-classification (JDC) model, using all dev set for training, eva set for evaluation
          Write out eer result each epoch
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
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import Adam
import hat.objectives as obj
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data
import prepare_eva_data as pp_eva_data
import theano.tensor as T
import cb_eer
from main_dev_jdc import loss_func, mean_pool

# hyper-params
dev_fe_fd = cfg.dev_fe_mel_fd
eva_fe_fd = cfg.eva_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len
act = 'relu'
n_hid = 500
fold = 0
n_out = len( cfg.labels )
    
    
# create empty folders in workspace
def create_folders():
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Md_eva_jdc' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_eva' )
    pp_dev_data.CreateFolder( cfg.scrap_fd + '/Results_eva/jdc_eer' )


# train model
def train():
    
    # create empty folders in workspace
    create_folders()
    
    # get dev & eva data
    tr_X, tr_y, _, _, _, _ = pp_dev_data.GetSegData( dev_fe_fd, agg_num, hop, fold=None )
    te_X, te_na_list = pp_eva_data.GetEvaSegData( eva_fe_fd, agg_num, hop )

    [n_songs, n_chunks, _, n_in] = tr_X.shape    
    print tr_X.shape, tr_y.shape
    print te_X.shape

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
    dump_fd = cfg.scrap_fd + '/Results_eva/jdc_eer'
    print_scores = cb_eer.PrintScoresBagOfBlocks( te_X, te_na_list, dump_fd, call_freq=1 )
    
    # callback, print loss each epoch
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, 
                            metrics=[loss_func], call_freq=1, dump_path=None )
                       
    # callback, save model every N epochs
    save_model = SaveModel( dump_fd=cfg.scrap_fd+'/Md_eva_jdc', call_freq=10 )
    
    # combine all callbacks
    callbacks = [ validation, save_model, print_scores ]
    
    # optimizer
    optimizer = Adam(2e-4)
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=10, n_epochs=1001, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks )
    
    
if __name__ == '__main__':
    train()