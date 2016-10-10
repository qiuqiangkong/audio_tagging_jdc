'''
SUMMARY:  plot results (must run after running all folds for main_dev_bob.py, main_dev_jdc.py 
          and main_eva_bob.py, main_eva_jdc.py)
AUTHOR:   Qiuqiang Kong
Created:  2016.09.12
Modified: 2016.10.09 update
--------------------------------------
'''
import pickle
import matplotlib.pyplot as plt
import config as cfg
import numpy as np


epoch = 10
t = np.arange( epoch )

def plot_errs_shadow( pickle_fd ):
    eers_mat = []
    for fold in xrange(5):
        errs = pickle.load( open( pickle_fd + '/eer_fold' + str(fold) + '.p', 'rb' ) )
        eers_mat.append( errs )
        
    eers_mat = np.array( eers_mat )
    eers_mat = eers_mat.T
    
    eers_mean = np.mean( eers_mat, axis=1 )
    eers_std = np.std( eers_mat, axis=1 )
    
    low_bound = eers_mean-eers_std
    up_bound = eers_mean+eers_std
    
    return eers_mean, low_bound, up_bound
    

alpha_plot = 0.3
alpha_between = 0.2
    
# bag of blocks dev
pickle_file = cfg.scrap_fd + '/Results_dev/bob_eer'
bob_dev_mean, bob_low_bound, bob_up_bound = plot_errs_shadow( pickle_file )

line_bob_dev, = plt.plot( t, bob_dev_mean, color='b', alpha=alpha_plot, label='dev bob' )
plt.fill_between( t, bob_low_bound, bob_up_bound, facecolor='blue', interpolate=True, alpha=alpha_between )

# joint detection classification dev
pickle_file = cfg.scrap_fd + '/Results_dev/jdc_eer'
jdc_dev_mean, jdc_low_bound, jdc_up_bound = plot_errs_shadow( pickle_file )

line_jdc_dev, = plt.plot( t, jdc_dev_mean, color='r', alpha=alpha_plot, label='dev jdc' )
plt.fill_between( t, jdc_low_bound, jdc_up_bound, facecolor='red', interpolate=True, alpha=alpha_between )


# bob eva
bob_eva_eers = pickle.load( open( cfg.scrap_fd + '/Results_eva/bob_eer.p', 'rb' ) )
line_bob_eva, = plt.plot(bob_eva_eers, color='b', label='eva bob')

# jdc eva
jdc_eva_eers = pickle.load( open( cfg.scrap_fd + '/Results_eva/jdc_eer.p', 'rb' ) )
line_jdc_eva, = plt.plot(jdc_eva_eers, color='r', label='eva jdc')

# plot all
plt.axis([0, epoch, 0, 0.5])
plt.xlabel('epoch')
plt.ylabel('EER')
plt.legend( handles=[line_bob_dev, line_jdc_dev, line_bob_eva, line_jdc_eva] )
plt.show()