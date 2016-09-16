'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: 2016.09.13 Add annotation
--------------------------------------
'''

### paths configuration

# development set
dev_wav_fd = '/vol/vssp/datasets/audio/dcase2016/chime_home/chunks'
dev_anno_fd = dev_wav_fd
dev_cv_csv_path = '/vol/vssp/datasets/audio/dcase2016/chime_home/development_chunks_refined_crossval_dcase2016.csv'

# evaluation set
eva_wav_fd = '/vol/vssp/AP_datasets/audio/dcase2016/task4/chime_home/eva_chunks'
eva_anno_fd = '/vol/vssp/AP_datasets/audio/dcase2016/task4/chime_home/chunk_annotations/annotations'
eva_csv_path = '/vol/vssp/AP_datasets/audio/dcase2016/task4/chime_home/chunk_annotations/evaluation_chunks_refined.csv'

# your workspace
scrap_fd = "/vol/vssp/msos/qk/DCASE2016_task4_scrap"
dev_fe_fd = scrap_fd + '/Fe_dev'
dev_fe_mel_fd = dev_fe_fd + '/Mel'
eva_fe_fd = scrap_fd + '/Fe_eva'
eva_fe_mel_fd = eva_fe_fd + '/Mel'


### global configuration

# labels
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.     # sample rate
win = 1024.     # fft window size
n_folds = 5