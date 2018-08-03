
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from pomegranate import *


# In[2]:


def collect_files(person_folders, action_folders, dir_path = os.getcwd()):
    # person_folders : an array of strings representing person's folder names
    # action : an array of strings representing action/sentence folder names
    # dirpath : path of root folder, defaults to present working directory
    
    # return type : list of file-paths
    
    if not os.path.isdir(dir_path):
        raise Exception("error: Specified project directory does not exist! Directory = " + dir_path)
    
    all_files = []
    for action in action_folders:
        action_files = []
        for person in person_folders:
            dirpath = dir_path + '\\group15\\group15 segmented\\'+ person +'\\' + action + '\\'    
            if not os.path.exists(dirpath):
                raise Exception("error: Specified action directory does not exist! Directory = " + dirpath)
            person_files = [(dirpath+f) for f in os.listdir(dirpath) if f.endswith('csv')]
            action_files.extend(person_files)
        all_files.extend(action_files)

    # print (all_files)
    
    return all_files

def csv_to_nparray(files, with_labels = False):
    # files : an array of absolute csv file locations to be read
    # with_labels : if true, the last column is read into a separate array, representing labels
    
    # return type : list of numpy ndarrays, and list of corrosponding labels if with_labels=True
    
    # prepare column names to extract
    col_names = []
    col_names.extend(['ALX','ALY','ALZ','ARX','ARY','ARZ'])
    col_names.extend(['EMG0L','EMG1L','EMG2L','EMG3L','EMG4L','EMG5L','EMG6L','EMG7L'])
    col_names.extend(['EMG0R','EMG1R','EMG2R','EMG3R','EMG4R','EMG5R','EMG6R','EMG7R'])
    col_names.extend(['GLX','GLY','GLZ','GRX','GRY','GRZ'])
    col_names.extend(['ORL','OPL','OYL','ORR','OPR','OYR'])
#     col_names.extend(['LHNX','LHNY','LHNZ','RHNX','RHNY','RHNZ'])
#     col_names.extend(['HX','HY','HZ'])
#     col_names.extend(['LTX','LTY','LTZ','RTX','RTY','RTZ'])
#     col_names.extend(['SpineMidX','SpineMidY','SpineMidZ'])
#     col_names.extend(['SpineBaseX','SpineBaseY','SpineBaseZ'])
#     col_names.extend(['ShoulderLeftX','ShoulderLeftY','ShoulderLeftZ','ShoulderRightX','ShoulderRightY','ShoulderRightZ'])
#     col_names.extend(['ElbowLeftX','ElbowLeftY','ElbowLeftZ','ElbowRightX','ElbowRightY','ElbowRightZ'])
    
    # define data types for each column name
    col_dtypes = {}
    for col_name in col_names:
        col_dtypes[col_name] = np.float64
    
    use_cols = col_names[:]
    instances = []
    
    if with_labels:
        label_col_name = 'state'
        use_cols.append(label_col_name)
        col_dtypes[label_col_name] = np.int32
        state_labels = []
    
    for file in files:
        # read csv into pandas dataframe
        df = pd.read_csv(file, index_col = False, usecols = use_cols, dtype = col_dtypes)
        # collect instance data
        instance = df.as_matrix(col_names)
        instances.append(instance)
        if with_labels:
            # collect state labels
            state_label = df.get(label_col_name).values
            state_labels.append(state_label)
    
    if with_labels:
        return instances, state_labels
    else:
        return instances

def decode_sequence(seq, decoder = None):
    # filter out contiguous repeating elements
    seq_ = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            seq_.append(seq[i])
    # decode
    if decoder is not None:
        decoded = []
        for code in seq_:
            if code in decoder:
                decoded.append(decoder[code])
            else:
                decoded.append('nan')
        return decoded
    else:
        return seq_

def fft_transform(dataset):
    fft_res = []
    for data in dataset:
        t_data = np.transpose(data)
        new_tdata = []
        for row in t_data:
            new_tdata.append(np.fft.fft(row))
        fft_res.append(np.transpose(new_tdata))
    return fft_res
    
def concatenate_instances(instances):
    ret_array = instances[0]
    lengths = [len(instances[0])]
    
    for i in range(1, len(instances)):
        ret_array = np.concatenate((ret_array, instances[i]))
        lengths.append(len(instances[i]))
    
    return ret_array, lengths

def check_segmentation(sequence):
    prev = sequence[0]
    n_components = 1
    transition_after = []
    for i in range(1, len(sequence)):
        if (sequence[i-1] != sequence[i]):
            # state transition detected
            n_components += 1
            transition_after.append(i-1)
    return n_components, transition_after


# In[3]:


sentences = ['S1']
decoder = {0 : '.', 1 : 'good night', 2 : 'father', 3 : 'and', 4 : 'cop'}

train_persons = ['samarth', 'hena', 'nishtha']
train_files = collect_files(train_persons, sentences)

test_persons = ['achal']
test_files = collect_files(test_persons, sentences)

train_instances = csv_to_nparray(train_files)
test_instances = csv_to_nparray(test_files)

train_instances = fft_transform(train_instances)
test_instances = fft_transform(test_instances)

# instances, lengths = concatenate_instances(train_instances)


# In[4]:


components = 5

# trans_mat = [[0.0 for j in range(components)] for i in range(components)]

# for i in range(components-1):
#     trans_mat[i][i] = 0.6
#     trans_mat[i][i+1]=0.4
# trans_mat[components-1][components-1] = 1.0

# starts = [0.0 for i in range(components)]
# starts[0] = 1.0
# ends = [0.0 for i in range(components)]
# ends[components-1] = 1.0

# model = hmm.GaussianHMM(\
#                         n_components=components, \
#                         covariance_type='diag', \
#                         algorithm='viterbi', \
#                         params='cmt', \
#                         init_params='cm', \
#                         n_iter=10000, \
#                         verbose=True\
#                        )

# model.startprob_ = starts
# model.transmat_ = trans_mat

# model.fit(instances, lengths)

# nd_list = []
# for i in range(len(train_instances[0][0])):
#     nd = NormalDistribution(1, 0)
#     nd_list.append(nd)

# i_dist = IndependentComponentsDistribution(nd_list)

# states = [State(i_dist) for i in range(components)]
# model = HiddenMarkovModel()
# for i in range(components):
#     model.add_state(states[i])
# for i in range(components-1):
#     model.add_transition(states[i], states[i], 0.1)
#     model.add_transition(states[i], states[i+1], 0.9)

# model.add_transition(states[components-1], states[components-1], 1.0)
# model.add_transition(model.start, states[0], 1.0)

# model.bake()

# model.fit(train_instances)

# dis = MultivariateGaussianDistribution(np.array([0.0 for i in range(len(train_instances[0][0]))]), \
#                                      np.eye(len(train_instances[0][0])))

# dists = [dis for i in range(components)]

model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=components, X=train_instances ,verbose=True)


# In[5]:


i = 0
for instance in test_instances:
    # predict
    seq = model.predict(instance)
    p_sen = ''.join([s+' ' for s in decode_sequence(seq, decoder)])
    
    # display
    print('[', i+1, ']')
    print('predicted: ', p_sen)
    print('actual: ', "good night father and cop .")
    print('')
    
    i += 1
    

