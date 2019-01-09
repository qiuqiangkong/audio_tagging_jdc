'''
SUMMARY:  callback function to write out scores every epoch
          This file is inherited from hat package's callback
          (callback is used to do something after 1 or several training epoch)
AUTHOR:   Qiuqiang Kong
Created:  2016.09.12
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.callbacks import Callback
import hat.backend as K
from hat.supports import to_list
import config as cfg
import numpy as np


# callback for the JDC model, write out all songs' all tags' score to txt every epoch 
class PrintScoresBagOfBlocks( Callback ):
    def __init__( self, te_x, te_na_list, dump_fd, batch_size=100, call_freq=1 ):
        self._call_freq_ = call_freq
        self._dump_fd_ = dump_fd
        self._te_x_ = K.format_data(te_x)   # shape: (n_songs, n_chunk, agg_num, n_in)
        self._te_na_list_ = te_na_list      # length: n_songs
        
    def compile( self, md ):
        self._md_ = md
        input_nodes = md.in_nodes_
        self._f_pred = K.function_no_given( input_nodes + [md.tr_phase_node_], md.out_nodes_ )
        
    def call( self ):
        y = self._f_pred( self._te_x_, 0. )     # recognize phase
        out = y[0]      # shape: (n_songs, n_chunks, n_out)
        p_y_pred = np.mean( out, axis=1 )   # shape: (n_songs, n_out)
        
        # write out results to txt file
        txt_path = self._dump_fd_+'/'+str(self._md_.epoch_)+'epoch.txt'
        print txt_path
        fwrite = open( txt_path, 'w' )
        for i1 in xrange( len(p_y_pred) ):
            for j1 in xrange(7):
                fwrite.write( self._te_na_list_[i1]+'.16kHz.wav' + ',' + cfg.id_to_lb[j1] + ',' + str(p_y_pred[i1][j1]) + '\n' )
        
        fwrite.close()
        
        
        
# callback for the JDC model, write out all songs' all tags' score to txt every epoch        
class PrintScoresDetectionClassification( Callback ):
    def __init__( self, te_x, te_na_list, dump_fd, batch_size=100, call_freq=1
                ):
        self._call_freq_ = call_freq
        self._dump_fd_ = dump_fd
        self._te_x_ = K.format_data(te_x)   # shape: (n_songs, n_chunk, agg_num, n_in)
        self._te_na_list_ = te_na_list      # length: n_songs
        
    def compile( self, md ):
        input_nodes = md.in_nodes_
        self._f_pred = K.function_no_given( input_nodes + [md.tr_phase_node_], md.out_nodes_ )
        self._md_ = md
        
    def call( self ):
        [out, mask] = self._f_pred( self._te_x_, 0. )     # recognize phase
                                                          # out & mask's shape: (n_songs, n_chunks, n_out)
        
        # final probability of each tag on each song, shape: (n_songs, n_labels)
        weighted_out = np.sum( out*mask / np.sum( mask, axis=1 )[:,None,:], axis=1 )
        
        # write out results to txt file
        fwrite = open(self._dump_fd_+'/'+str(self._md_.epoch_)+'epoch.txt', 'w')
        print self._dump_fd_+'/'+str(self._md_.epoch_)+'epoch.txt'
        for i1 in xrange( len(weighted_out) ):
            for j1 in xrange(7):
                fwrite.write( self._te_na_list_[i1]+'.16kHz.wav' + ',' + cfg.id_to_lb[j1] + ',' + str(weighted_out[i1][j1]) + '\n' )
        
        fwrite.close()