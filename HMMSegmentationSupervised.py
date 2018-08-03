
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from pomegranate import *
import time


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


# In[3]:


sentences = ['S1']
decoder = {0 : '.', 1 : 'good night', 2 : 'father', 3 : 'and', 4 : 'cop'}

train_persons = ['samarth', 'hena', 'nishtha']
test_persons = ['achal']

train_files = collect_files(train_persons, sentences)
test_files = collect_files(test_persons, sentences)

train_instances, train_labels = csv_to_nparray(train_files, with_labels=True)
test_instances, test_labels = csv_to_nparray(test_files, with_labels=True)

train_instances = fft_transform(train_instances)
test_instances = fft_transform(test_instances)


# prepare the model parameters

print('training the HMM model...')
t1 = time.time()

# number of HMM states
components = 5

# prepare start probability vector
starts = [0.0 for i in range(components)]
starts[1] = 1.0

# prepare state transition matrix
trans_mat = [[0.0 for j in range(5)] for i in range(components)]
trans_mat[0][0] = 1.0
train_labels_ = np.concatenate(train_labels)
for i in range(1, components):
    j = i+1
    if j == components:
        j = 0
    i_labels = [l for l in train_labels_ if l == i]
    n = float(len(i_labels))
    trans_mat[i][i] = n / (n+1.0)
    trans_mat[i][j] = 1.0 / (n+1.0)

# train conditional Gaussian distributions
train_instances_ = np.concatenate(train_instances)
dists = []
label_inst = train_instances_[train_labels_ == 0]
dists = []
for i in range(components):
    label_instances = train_instances_[train_labels_ == i]
    d = MultivariateGaussianDistribution.from_samples(label_instances)
    dists.append(d)

# initialize the HMM
model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts)
train_time = time.time() - t1
print('...HMM model trained in ', train_time, 'seconds.')
print('')



acc_train = 0.0
n_train = float(len(train_instances))

acc_test = 0.0
n_test = float(len(test_instances))

print('----------------------------')
print('  TRAINING SET PRECICTIONS: ')
print('----------------------------')
t1 = time.time()
for i in range(len(train_instances)):
    p_sen = ''.join([s+' ' for s in decode_sequence(model.predict(train_instances[i]), decoder)])
    a_sen = ''.join([s+' ' for s in decode_sequence(train_labels[i], decoder)])
    if p_sen == a_sen: acc_train += 1
    print('[', i+1, ']')
    print('predicted: ', p_sen)
    print('actual: ', a_sen)
    print('')

print('')

print('----------------------------')
print('  TEST SET PREDICTIONS: ')
print('----------------------------')
for i in range(len(test_instances)):
    p_sen = ''.join([s+' ' for s in decode_sequence(model.predict(test_instances[i]), decoder)])
    a_sen = ''.join([s+' ' for s in decode_sequence(test_labels[i], decoder)])
    if p_sen == a_sen: acc_test += 1
    print('[', i+1, ']')
    print('predicted: ', p_sen)
    print('actual: ', a_sen)
    print('')
pred_time = (time.time() - t1) / float(n_train+n_test)

print('------------')
print('  SUMMARY: ')
print('------------')

acc_train /= n_train
acc_test /= n_test
print('Training set accuracy: ', acc_train * 100, '%')
print('Test set accuracy: ', acc_test * 100, '%')
print('Time to train the model: ', train_time, 'seconds')
print('Average prediction time: ', pred_time, 'seconds')
print('')

