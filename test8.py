# adaptive weight chunck recognize
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
import matplotlib.pyplot as plt
from main_dev_bob import fold
import time
import eer


# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len

# load model
md = serializations.load( cfg.scrap_fd + '/Md_dev_bob/md100.p' )

def reshape_3d_to_4d( X ):
    return X.reshape( (1,)+X.shape )
    
def reshape_3d_to_2d( X ):
    return X.reshape( X.shape[1:] )

def get_y_pred( out, mask ):
    weighted_out = np.sum( out*mask / np.sum( mask, axis=1 )[:,None,:], axis=1 )
    return weighted_out
    

def recognize():
    
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            if fold==curr_fold:
                # get features, tags
                
                na = 'CR_lounge_220110_0731.s0_chunk39'
                
                fe_path = fe_fd + '/' + na + '.f'
                info_path = cfg.dev_wav_fd + '/' + na + '.csv'
                tags = pp_data.GetTags( info_path )
                X = cPickle.load( open( fe_path, 'rb' ) )
                
                # aggregate data
                X3d = mat_2d_to_3d( X, agg_num, hop )
                X4d = reshape_3d_to_4d( X3d )
        
                p_y_pred = md.predict( X4d )    # shape: 1*n_chunks*agg_num*n_in
                tmp = p_y_pred
                p_y_pred = np.mean( reshape_3d_to_2d(p_y_pred), axis=0 )     # shape:(n_label)
                
                
                
                print na
                print tags
                fig, axs = plt.subplots(2,1, sharex=True)
                axs[0].matshow(np.log(X.T), origin='lower', aspect='auto')
                axs[1].matshow(tmp[0].T, origin='lower', aspect='auto')
                
                axs[0].get_xaxis().set_visible(False)
                axs[1].xaxis.set_ticks_position('bottom')
                axs[1].set_yticklabels([''] + cfg.labels)
                axs[0].set_title('log Mel spectrogram')
                axs[1].set_title('classifier output')
                plt.show()
                pause
    

if __name__ == '__main__':
    recognize()