'''
Created on Feb 1, 2017

@author: atrisha
'''
import get_a_batch

def get_batch():
    #data_batch = tf.placeholder(tf.float32, [BATCH_SIZE,GRID_SIZE,GRID_SIZE,4])
    #data_target = tf.placeholder(tf.float32, [BATCH_SIZE,GRID_SIZE,GRID_SIZE,3])
    '''
    following dimensions are expected
    env_data = batch_size X 5 X 10 X 6
    ego_data = batch_size X 6 X 1
    target_data = batch_size X 2 X 1
    '''
    #print('called train batch')
    env_batch,ego_batch,target_batch,(max_index,density) = get_a_batch.get_a_data_batch()
    #print('returned train batch')
    return env_batch,ego_batch,target_batch,(max_index,density)

def get_evaluation_batch(for_dbn):
    print('called eval batch')
    env_batch,ego_batch,target_batch,_ = get_a_batch.get_a_evaluation_data_batch(for_dbn)
    print('returned eval batch')
    return env_batch,ego_batch,target_batch,_

