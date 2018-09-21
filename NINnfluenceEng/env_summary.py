'''
Created on Feb 6, 2017

@author: atrisha
'''
import nn_engine_input
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as sp
from mpl_toolkits.mplot3d import axes3d

'''
cl_id : { 1 : {[x_min,x_max] , [y_min , y_max]} , 2 :{ [..] , [...]}
'''

cluster_segments_map = { 0: { 1: [[-14,6.2] , [41.97,58.7]] , 2: [[-14,6.2] , [29.42,41.97]] , 3: [[-14,6.2] , [16.74,29.42]], 4: [[-14,6.2] , [7.77,16.74]] }  ,
                            1: {1: [[2.86,6.8] , [8.54,16.7]], 2: [[-0.73,2.86] , [8.54,16.7]], 3:[[-3.44,-0.73] , [8.54,16.7]], 4:[[-8,-3.44] , [8.54,16.7]]} ,
                                2: { 1: [[-10.8,-5] , [43.22,27.75]], 2: [[-10.8,-5] ,[27.75,19.18]], 3: [[-10.8,-5] , [19.18,14.70]], 4: [[-10.8,-5] , [14.70,12.82]] },
                                    3: { 1: [[-8.2,-3.6] , [19,16.25]], 2: [[-8.2,-3.6] , [16.25,13.5]], 3: [[-8.2,-3.6] ,[13.5,10.75]], 4: [[-8.2,-3.6] , [10.75,8]] },
                                        4: { 1: [[-8,-4.3] ,[8.5,10.75]], 2: [[-8,-4.3] , [10.75,13]], 3: [[-8,-4.3] , [13,15.25]], 4: [[-8,-4.3] , [15.25,17.5]] }
                        }

alpha = 0.6
beta = 0.1


def get_influence_vector(cluster_id , cluster_details_vector):
    ''' cluster_segment_details = { 1(segment_key) : [num_agents,sum_velocity]} .... '''
    cluster_segment_details = dict()
    segments = cluster_segments_map[cluster_id]
    for agent in cluster_details_vector:
        x,y,x_v,y_v = agent[0],agent[1],agent[2],agent[3]
        if x !=0 or y!=0 or x_v !=0 or y_v !=0:
            for k,v in segments.items():
                if v[0][0] <= x <= v[0][1] and v[1][0] <= y <= v[1][1] :
                    if k in cluster_segment_details.keys():
                        cluster_segment_details[k][0] = cluster_segment_details[k][0] + 1
                        cluster_segment_details[k][1] = cluster_segment_details[k][1] + math.sqrt(x_v ** 2 + y_v ** 2)
                    else:
                        cluster_segment_details[k] = [1,math.sqrt(x_v ** 2 + y_v ** 2)]
    influence_vector = np.zeros(shape = (1,4))
    for i in range(1,influence_vector.shape[1]+1):
        if i in cluster_segment_details.keys():
            influence_vector[0,i-1] = alpha * cluster_segment_details[i][0] + beta*cluster_segment_details[i][1]
    return influence_vector 
        
        

def encode_env(env_slice):
    env_vector_l = []
    t = []
    for cluster_id in range(env_slice.shape[0]):
        influence_vector = get_influence_vector(cluster_id,env_slice[cluster_id])
        env_vector_l.append(influence_vector)
    env_vector = np.reshape(np.asarray(env_vector_l) , newshape = (1,20))
    return env_vector
    
def get_env(evaluation):
    if evaluation:
        env_data_batch, ego_data_batch,target_data_batch,(max_density_index,density) = nn_engine_input.get_evaluation_batch(False)
    else:
        env_data_batch, ego_data_batch,target_data_batch,(max_density_index,density) = nn_engine_input.get_batch()
    #x_max,x_min,y_max,y_min = 20,-20,55,-5
    #env_slice = env_data_batch[max_density_index]
    env_array = np.zeros(shape=(env_data_batch.shape[0],1,20))
    
    for step in range(env_data_batch.shape[0]):
        env_slice = env_data_batch[step]
        env_vector = encode_env(env_slice)
        env_array[step] = np.copy(env_vector)
    return env_array,ego_data_batch,target_data_batch  

