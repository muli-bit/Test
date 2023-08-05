import json
import time
import numpy as np
import sys
import os
import scipy.io
from error import error

from preprocessing import preprocessing_blocks
from preprocessing_function import filter_f

root_path = 'py_code//'
sys.path.append(root_path)

try:
    import mne  # 验证是否安装mne
except ImportError:
    from os import system

    system('pip install mne==1.0.3')
    import mne


# ############################### input parameters ######################
operation = 0 # 0:preprocess data by blocks; 1: get mae error between manual-clean and online-clean
# used when operation = 0
file_all = 'E:\WorkPlace\database\loreal\\0207\\'  # file to process
path_save = '../result/'
# used when operation = 1
manual_dir = 'E:\WorkPlace\database\loreal\预处理结果\\' # directory for manual preprocessing of files
auto_dir = '../result//' # directory for auto preprocessing of files


# ############################### initialize ######################
tf = open('../conf_block.json', 'r')
conf_block = json.load(tf)
srate = conf_block['srate']
ch_names = conf_block['channel_names']
ref_ch = conf_block['ref_ch']
EOG_ch = conf_block['EOG_ch']
block_time = conf_block['block_time']
# block_time = 2
# window = conf_block['window']
locs_path = conf_block['position']

# get ref_index
ref_idx = []
for ref in ref_ch:
    ref_idx.append(np.where(np.array(ch_names) == ref)[0][0])
ref_idx = np.sort(ref_idx)  # ensure the increasing order
# get eog_index
eog_idx = []
for eog in EOG_ch:
    eog_idx.append(np.where(np.array(ch_names) == eog)[0][0])
eog_idx = np.sort(eog_idx)

montage = mne.channels.read_custom_montage(locs_path, head_size=0.085)

picks = ch_names.copy()
# remove EOG
for i in range(len(ref_ch)):
    picks.remove(ref_ch[i])
picks_save = picks.copy()
for i in range(len(EOG_ch)):
    picks.remove(EOG_ch[i])

# get the position
pos_dict = montage.get_positions()['ch_pos']
pos = []
for i in picks:
    pos.append(pos_dict[i].tolist())
pos = np.array(pos)

# design the filter
# hp_filter = filter_f(srate, 'highpass', l_freq=0.5, h_freq=None)
hp_filter = mne.filter.create_filter(None, sfreq=500, l_freq=0.5, h_freq=None, phase='minimum')
bs_filter = filter_f(srate, 'bandstop', l_freq=48, h_freq=52)
z1 = np.zeros([len(picks_save), len(hp_filter) - 1], dtype=float) # initial filter state
z2 = np.zeros([len(picks_save), len(bs_filter) - 1], dtype=float)
z_list=[z1,z2]
bp_filter = mne.filter.create_filter(None, sfreq=500, l_freq=1, h_freq=100, phase='minimum')
z_bp=np.zeros([len(picks), len(bp_filter) - 1], dtype=float) # used in ica step
ch_state=[0,0,0]
seg_state=[0,0,0]
block_points = int(block_time * srate)
data_before = np.ones([len(picks_save), block_points])

#################################### initialize recursive ica #####################################
# setting up the forgetting factors]
adaptiveFF = {'profile': 'constant', 'tau_const': np.Inf, 'gamma': 0.6, 'lambda_0': 0.995, 'decayRateAlpha': 0.02,
              'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}  #
# Evaluate convergence such as Non-Stationarity Index (NSI).
evalConvergence = {'profile': False, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}
block_size = block_points
numSubgaussian = 0
verbose = 1
nChs = len(picks)
state = {'icaweights': np.eye(nChs), 'icasphere': np.eye(nChs), 'lambda_k': np.zeros([1, block_size]),
         'minNonStatIdx': [], 'counter': 1}
if np.logical_or(adaptiveFF['profile'] == 'cooling', adaptiveFF['profile'] == 'constant'):
    adaptiveFF['lambda_const'] = 1 - np.exp(-1 / adaptiveFF['tau_const'])
if evalConvergence['profile']:
    state['Rn'] = []
    state['nonStatIdx'] = []
state['kurtsign'] = np.ones(nChs) > 0
if numSubgaussian != 0:
    state['kurtsign'][:numSubgaussian] = False


file_list = []
def list_files(path):
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    if dirs:
        for i in dirs:
            list_files(os.path.join(path, i))
    files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
    for f in files:
        file_list.append(os.path.join(path, f))
    return file_list


# #######################  simulate the input process ###########################
if operation == 0:  # preprocessing
    time0 = time.time()
    total_points = 0
    file_list_all = list_files(file_all)  # recursively list all the files

    for file in file_list_all:
        if file.endswith('vhdr'):
            path_vhdr=file
            path_vmrk = path_vhdr[:-len('.vhdr')]+ '.vmrk'
            raw = mne.io.read_raw_brainvision(path_vhdr, preload=True, scale=1e6)

            import re
            id0 = re.split("[/\\\]", path_vhdr)[-1][:-len('.vhdr')] # get person id
            data=raw.get_data()
            total_points += data.shape[1]

            for i in range(int(data.shape[1] / block_points)):
                data_block = data[:, i * block_points:(i + 1) * block_points]

                data_range = np.arange(i * block_points, (i + 1) * block_points)
                data_out, z_list, z_bp, ch_state, seg_state, bad, block_save, state = preprocessing_blocks(
                    data_block, ch_names, ref_ch, ref_idx,
                    eog_idx, picks, pos, filter_list=[hp_filter, bs_filter],
                    bp_filter=bp_filter, z_list=z_list, z_bp=z_bp,
                    ch_state=ch_state, seg_state=seg_state, state=state, dataRange=data_range,
                    adaptiveFF=adaptiveFF, evalConvergence=evalConvergence)
                if bad:
                    data_out = data_before
                data_before = data_out
                if i == 0:
                    result = data_out
                else:
                    result = np.concatenate([result, data_out], axis=1)

            result0 = {'data': result}
            scipy.io.savemat(path_save + id0 + '_EEG_cleaned_new.mat', result0)

    time1 = time.time()
    print("============================ time cost ===========================")
    sample_time = total_points / 500
    print("processed " + str(sample_time) + "s")
    print("total time:" + str(time1 - time0) + ' s')
    print("average time:" + str((time1 - time0) / (sample_time / block_time)) + 's (/block)\n')

if operation == 1:  # get error
    error(auto_dir, manual_dir)





