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
from main_dev_jdc import fold, mean_pool
import time
import eer
import wavio


# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

def reshape_3d_to_4d( X ):
    return X.reshape( (1,)+X.shape )
    
def reshape_3d_to_2d( X ):
    return X.reshape( X.shape[1:] )

def get_y_pred( out, mask ):
    weighted_out = np.sum( out*mask / np.sum( mask, axis=1 )[:,None,:], axis=1 )
    return weighted_out
    

def recognize():

    na = 'CR_lounge_220110_0731.s0_chunk39'
    
    fe_path = fe_fd + '/' + na + '.f'
    info_path = cfg.dev_wav_fd + '/' + na + '.csv'
    tags = pp_data.GetTags( info_path )
    X = cPickle.load( open( fe_path, 'rb' ) )
    wav,fs = readwav( cfg.dev_wav_fd + '/' + na + '.16kHz.wav' )
    wav = wav[0:int(41./52.*64000)]
    
    
    plt.plot(wav)
    plt.xticks( np.arange(0,int(41./52.*64000), 16000), [0,1,2,3] )
    plt.axis([0, int(41./52.*64000), -0.12, 0.12])
    plt.ylabel('amplitude')
    ax = plt.gca()
    ax.set_xlabel('s')
    ax.xaxis.set_label_coords(1.06, -0.01)
    
    
    # aggregate data
    X3d = mat_2d_to_3d( X, agg_num, hop )
    X4d = reshape_3d_to_4d( X3d )

    md = serializations.load( cfg.scrap_fd + '/Md_dev_jdc/md100.p' )
    [out, mask] = md.predict( X4d )    # shape: 1*n_chunks*agg_num*n_in
    p_y_pred = get_y_pred( out, mask ).flatten()
    
    
    
    print na
    print tags
    uni_mask = mask[0] / np.sum( mask[0], axis=0 )
    fig, axs = plt.subplots(6,1, sharex=False)
    axs[0].matshow(np.log(X.T), origin='lower', aspect='auto')
    axs[1].matshow(uni_mask.T, origin='lower', aspect='auto')
    axs[2].matshow(out[0].T, origin='lower', aspect='auto')
    axs[3].matshow( ( mask[0]*out[0] / np.sum(mask[0],axis=0) ).T, origin='lower', aspect='auto')
    
    #
    md = serializations.load( cfg.scrap_fd + '/Md_dev_bob/md100.p' )
    X3d = mat_2d_to_3d( X, agg_num, hop )
    X4d = reshape_3d_to_4d( X3d )

    p_y_pred = md.predict( X4d )    # shape: 1*n_chunks*agg_num*n_in
    tmp = p_y_pred
    p_y_pred = np.mean( reshape_3d_to_2d(p_y_pred), axis=0 )     # shape:(n_label)
    
    axs[4].matshow(tmp[0].T, origin='lower', aspect='auto')
  
    #
    
    gt = np.zeros(out[0].shape)
    '''
    gt[0:int(52*28604./64000.), 0] = 1
    #gt[int(52*52400./64000.):, 0] = 1
    gt[int(52*5252/64000.):int(52*58922./64000), 4] = 1
    gt[int(52*32182./64000):int(52*37491./64000.), 6] = 1
    gt[int(52*41820./64000.):int(52*48687./64000.), 6] = 1
    '''
    
    axs[5].matshow(np.log(gt).T, origin='lower', aspect='auto')
    
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)
    axs[3].get_xaxis().set_visible(False)
    axs[4].get_xaxis().set_visible(False)
    axs[5].xaxis.set_ticks_position('bottom')
    axs[1].set_yticklabels([''] + cfg.labels)
    axs[2].set_yticklabels([''] + cfg.labels)
    axs[3].set_yticklabels([''] + cfg.labels)
    axs[4].set_yticklabels([''] + cfg.labels)
    axs[5].set_yticklabels([''] + cfg.labels)
    axs[5].set_xlabel('frames')
    axs[5].xaxis.set_label_coords(1.06, -0.01)
    
    
    plt.show()
    pause
                
    

if __name__ == '__main__':
    recognize()