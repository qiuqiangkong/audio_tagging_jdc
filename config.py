'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: 
--------------------------------------
'''

# development
dev_root = '/vol/vssp/datasets/audio/dcase2016/chime_home'
dev_wav_fd = dev_root + '/chunks'

# temporary data folder
scrap_fd = "/vol/vssp/msos/qk/DCASE2016_task4_scrap"
dev_fe_mel_fd = scrap_fd + '/Fe/Mel'
dev_cv_csv_path = dev_root + '/development_chunks_refined_crossval_dcase2016.csv'

# evaluation
'''
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'
'''

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.
win = 1024.