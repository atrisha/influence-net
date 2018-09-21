'''
Created on Dec 5, 2016

@author: atrisha
'''
import db_helper as helper
import numpy as np
from trajectory_cluster import *

def sample(sampling_tuple , num_samples):
    num_scenarios = list(range(1,sampling_tuple[0]))
    num_scenarios.remove(sampling_tuple[1])
    scene_2_sample_from = np.random.choice(num_scenarios)
    db = helper.connect()
    cursor = db.cursor()
    string = "SELECT MIN(CAST(OBJECT.STARTFRAME AS UNSIGNED)),MAX(CAST(OBJECT.ENDFRAME AS UNSIGNED)) FROM OBJECT,ANNOTATION WHERE OBJECT.PID = ANNOTATION.ID AND ANNOTATION.ID =?"
    cursor.execute(string,[str(scene_2_sample_from)])
    res = cursor.fetchone()
    min_frame,max_frame = int(res[0]),int(res[1])
    db.close()    
    return np.random.randint(low = min_frame, high = max_frame, size = num_samples), scene_2_sample_from
    
def find_cluster_id(C,id):
    for key,value in C.items():
        if id in value:
            return key
    return -1

def get_tj_ids(frames_2_sample,scene_2_sample_from):
    db = helper.connect()
    cursor = db.cursor()
    tj_id_list,particle_id_list = [],[]
    for frames in frames_2_sample:
        string = "SELECT PT_CAMERA_COOR.ID,OBJECT.ID FROM PT_CAMERA_COOR,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND PT_CAMERA_COOR.T=? AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID=? ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
        cursor.execute(string,[str(frames),str(scene_2_sample_from)])
        res_1 = cursor.fetchone()
        id_list_in_frame,particle_list_in_frame = [],[]
        while res_1 is not None:
            id_list_in_frame.append(res_1[1])
            particle_list_in_frame.append(res_1[0])
            res_1 = cursor.fetchone()
        tj_id_list.append(id_list_in_frame)
        particle_id_list.append(particle_list_in_frame)
        
    db.close()
    return particle_id_list,tj_id_list

def id_2_particle(particle_id_list,merged_data_trajectory,merged_particle_ids,C,tj_id_list):
    particle_list = []
    ctr = 0
    for p in particle_id_list:
        cluster_id = find_cluster_id(C,int(tj_id_list[ctr]))
        ctr = ctr + 1
        ind = merged_particle_ids.index(p)
        particle_tuple = merged_data_trajectory[ind]
        '''particle_tuple = list(particle_tuple)'''
        '''particle_tuple.append(cluster_id)'''
        '''particle_tuple = tuple(particle_tuple)'''
        particle_list.append(particle_tuple)
    
    return particle_list

def particle_sample(sampling_tuple , merged_data_trajectory , merged_particle_ids , C ):
    frames_2_sample,scene_2_sample_from = sample(sampling_tuple,200)
    sampled_particle_ids,sample_tj_ids = get_tj_ids(frames_2_sample,scene_2_sample_from)
    particle_list = [id_2_particle(x,merged_data_trajectory,merged_particle_ids,C,y) for x,y in zip(sampled_particle_ids,sample_tj_ids)]
    return particle_list
