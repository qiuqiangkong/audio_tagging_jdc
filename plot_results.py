# 2016.09.10
import pickle
import matplotlib.pyplot as plt
import config as cfg
import numpy as np



epoch = 300
t = np.arange( epoch )

def plot_errs_shadow( pickle_fd ):
    eers_mat = []
    for fold in xrange(5):
        errs = pickle.load( open( pickle_fd + '/fold' + str(fold) + '.p', 'rb' ) )
        eers_mat.append( errs )
        
    eers_mat = np.array( eers_mat )
    eers_mat = eers_mat.T
    
    eers_mean = np.mean( eers_mat, axis=1 )
    eers_std = np.std( eers_mat, axis=1 )
    
    low_bound = eers_mean-eers_std
    up_bound = eers_mean+eers_std
    
    return eers_mean, low_bound, up_bound
    
    
def plot_errs_shadow2( pickle_fd ):
    eers_mat = []
    for fold in xrange(5):
        errs = pickle.load( open( pickle_fd + '/fold' + str(fold) + '.p', 'rb' ) )
        eers_mat.append( errs )
        
    eers_mat = np.array( eers_mat )
    eers_mat = eers_mat.T
    
    return eers_mat



alpha_plot = 0.3
alpha_between = 0.2
    
# bag of blocks dev
pickle_file = cfg.scrap_fd + '/Results/bagofblock_eer'
bob_dev_mean, bob_low_bound, bob_up_bound = plot_errs_shadow( pickle_file )

line_bob_dev, = plt.plot( t, bob_dev_mean, color='b', alpha=alpha_plot, label='dev bob' )
plt.fill_between( t, bob_low_bound, bob_up_bound, facecolor='blue', interpolate=True, alpha=alpha_between )

# joint detection classification dev
pickle_file = cfg.scrap_fd + '/Results/detectionclassification_eer'
jdc_dev_mean, jdc_low_bound, jdc_up_bound = plot_errs_shadow( pickle_file )

line_jdc_dev, = plt.plot( t, jdc_dev_mean, color='r', alpha=alpha_plot, label='dev jdc' )
plt.fill_between( t, jdc_low_bound, jdc_up_bound, facecolor='red', interpolate=True, alpha=alpha_between )


# bob eva
bob_eva_eers = pickle.load( open( cfg.scrap_fd + '/Results_eva/bagofblock_eer/eva.p', 'rb' ) )
line_bob_eva, = plt.plot(bob_eva_eers, color='b', label='eva bob')

# jdc eva
jdc_eva_eers = pickle.load( open( cfg.scrap_fd + '/Results_eva/detectionclassification_eer/eva.p', 'rb' ) )
line_jdc_eva, = plt.plot(jdc_eva_eers, color='r', label='eva jdc')


# bob tr
pickle_file = cfg.scrap_fd + '/Results_tr/bagofblock_eer'
bob_mean, bob_low_bound, bob_up_bound = plot_errs_shadow( pickle_file )

line_bob_tr, = plt.plot( t, bob_mean, '--', color='b', alpha=alpha_plot, label='tr bob' )
plt.fill_between( t, bob_low_bound, bob_up_bound, facecolor='blue', interpolate=True, alpha=alpha_between )

# jdc tr
pickle_file = cfg.scrap_fd + '/Results_tr/detectionclassification_eer'
jdc_mean, jdc_low_bound, jdc_up_bound = plot_errs_shadow( pickle_file )

line_jdc_tr, = plt.plot( t, jdc_mean, '--', color='r', alpha=alpha_plot, label='tr jdc' )
plt.fill_between( t, jdc_low_bound, jdc_up_bound, facecolor='red', interpolate=True, alpha=alpha_between )


# plot all
plt.axis([0, 300, 0, 0.5])
plt.xlabel('epoch')
plt.ylabel('EER')
plt.legend( handles=[line_bob_tr, line_jdc_tr, line_bob_dev, line_jdc_dev, line_bob_eva, line_jdc_eva] )
plt.show()

print bob_dev_mean[65], jdc_dev_mean[110]
print bob_eva_eers[65], jdc_eva_eers[110]


'''
alpha_plot = 0.3
alpha_between = 0.2
    
# bag of blocks dev
pickle_file = cfg.scrap_fd + '/Results/bagofblock_eer'
bob_mean, bob_low_bound, bob_up_bound = plot_errs_shadow( pickle_file )
bob_eers_mat = plot_errs_shadow2( pickle_file )

#line_bob_dev, = plt.plot( t, bob_mean, color='b', alpha=alpha_plot, label='bob_dev' )
#plt.fill_between( t, bob_low_bound, bob_up_bound, facecolor='blue', interpolate=True, alpha=alpha_between )

for i1 in xrange(5):
    plt.plot(bob_eers_mat[:,i1], 'b', alpha=(i1+1)*0.2)
    
# jdc dev    
pickle_file = cfg.scrap_fd + '/Results/detectionclassification_eer'
jdc_mean, jdc_low_bound, jdc_up_bound = plot_errs_shadow( pickle_file )
jdc_errs_mat = plot_errs_shadow2( pickle_file )

for i1 in xrange(5):
    plt.plot(jdc_errs_mat[:,i1], 'r', alpha=(i1+1)*0.2)
    
# plot all
plt.axis([0, 300, 0, 0.5])
#plt.legend( handles=[line_bob_dev, line_jdc_dev, line_bob_eva, line_bob_eva] )
plt.show()
'''