'''
Created on Jan 25, 2017

@author: atrisha
'''
import numpy as np
import db_helper as helper
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 40500
density_2_look_4 = 1

RANGE_X = 20
RANGE_Y = 50

RANGE_X_V = 430
RANGE_Y_V = 430

def get_a_data_batch():
    density_max = 0
    db = helper.connect()
    cursor = db.cursor()
    string = """SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,PT_CAMERA_COOR_ADD_OPTS.THETA,PT_CAMERA_COOR_ADD_OPTS.X_V,PT_CAMERA_COOR_ADD_OPTS.Y_V,PT_CAMERA_COOR_ADD_OPTS.CLASS,ANNOTATION.ID,OBJECT.ID,PT_CAMERA_COOR_ADD_OPTS.CLUSTER 
                    FROM PT_CAMERA_COOR,PT_CAMERA_COOR_ADD_OPTS,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID NOT IN ('8') AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID 
                        ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"""
    cursor.execute(string)
    results = cursor.fetchall()
    
    x = map(list, list(results))
    x = list(x)
    n=np.asarray(x,dtype=np.float32)
    db.close()
    env_batch = np.zeros(shape=(BATCH_SIZE,5,6,7), dtype = np.float32)
    env_batch_slice = np.zeros(shape=(5,6,7), dtype = np.float32)
    target_batch = np.zeros(shape=(BATCH_SIZE,2,1) , dtype = np.float32)
    ego_batch = np.zeros(shape=(BATCH_SIZE,7,1) , dtype = np.float32)
    '''here :env_batch = np.zeros(shape=(BATCH_SIZE,1,7), dtype = np.float32)
    here :env_batch_slice = np.zeros(shape=(1,7), dtype = np.float32)
    here :target_batch = np.zeros(shape=(BATCH_SIZE,2,1) , dtype = np.float32)
    here :ego_batch = np.zeros(shape=(BATCH_SIZE,7,1) , dtype = np.float32)'''
    ''' we need consecutive frames of the same annotation id having the same object ids
    once we have these 2 frames, we create a image grid and a target grid
    
    1. sample one scenario randomly between min and max but not in evaluation scenario
    2. get t_id range in scenario from 1. sample a t_id
    3. run 2 queries with (scenario id in 1 and t_id in 2) and (scenario id in 1 and t_id+1 in 2) 
    4. if list of object_ids in the result of both queries in 2 are same, use the result as consecutive frames. else from 2.
    '''
    fill_index = 0
    max_density_index = 0
    while fill_index < 20000:
        found_it = False
        while(not found_it):
            #grid_data = np.zeros(shape=(GRID_SIZE,GRID_SIZE,4))
            #grid_target = np.zeros(shape=(GRID_SIZE,GRID_SIZE,3))
            ''' Step 1'''
            min_s,max_s = np.amin(n[:,9]),np.amax(n[:,9])
            scenes = list(range(int(min_s),int(max_s)+1))
            scenes.remove(8)
            scenario_id = np.random.choice(scenes)
            ''' Step 2'''
            c1 = n[:,9]==scenario_id
            r = n[c1]
            min,max = np.amin(r[:,4]),np.amax(r[:,4])
            t_id = np.random.randint(min,max+1)
            ''' Step 3'''
            c2 = n[:,4]==t_id
            c3 = n[:,4]==t_id+1
            c4 = np.logical_and(c1,c2)
            c5 = np.logical_and(c1,c3)
            r1 = n[c4]
            r2 = n[c5]
            ''' Step 4'''
            object_ids_r1 = r1[:,10]
            object_ids_r2 = r2[:,10]
            '''if len(object_ids_r1) != density_2_look_4:
                found_it = False
                continue'''
            if np.array_equal(object_ids_r1, object_ids_r2):
                for o_ids in object_ids_r1:
                    env_batch_slice = np.zeros(shape=(5,6,6), dtype = np.float32)
                    '''here :env_batch_slice = np.zeros(shape=(1,6), dtype = np.float32)'''
                    #print('density is',len(object_ids_r1),' ego object id is',o_ids)
                    density_value = len(object_ids_r1)
                    if density_value > density_max:
                        density_max = density_value
                        max_density_index = fill_index
                    density_array = np.zeros(shape=(5,6,1), dtype = np.float32)
                    '''here :density_array = np.zeros(shape=(1,1), dtype = np.float32)'''
                    density_array.fill(density_value)
                    obj_ind = {0:0 , 1:0 , 2:0 , 3:0 , 4:0}
                    for i in range(r1.shape[0]):
                        if r1[i,10] == o_ids:
                            #print('ego obj id is',o_ids)
                            x,y = float(r1[i,2]) , float(r1[i,3])
                            fea_vector = np.reshape(np.asarray([x  , y  ,float(r1[i,6])  , float(r1[i,7])  , int(r1[i,8]),  int(r1[i,11]) , density_value]) , newshape = (7,1))
                            ego_batch[fill_index] = np.copy(fea_vector)
                            for i2 in range(r2.shape[0]):
                                if r2[i2,10] == o_ids:
                                    #print('target entered for obj id',o_ids)
                                    y_a = float(r2[i2,7]) - float(r1[i,7])
                                    x_a = float(r2[i2,6]) - float(r1[i,6])
                                    x_n,y_n = float(r2[i,2]) , float(r2[i,3])
                                    fea_vector = np.reshape(np.asarray([x_a,y_a]) , newshape = (2,1))
                                    target_batch[fill_index] = np.copy(fea_vector)  
                        else:
                            #print('added env entry for object id',r1[i,10])
                            x,y = float(r1[i,2]) , float(r1[i,3])
                            cluster_id = int(r1[i,11])
                            fea_vector = np.asarray([x,y,float(r1[i,6]) , float(r1[i,7]) , int(r1[i,8])+10,  int(r1[i,11])])
                            if cluster_id < 5:
                                env_batch_slice[cluster_id,obj_ind[cluster_id]] = np.copy(fea_vector)
                                '''here : env_batch_slice[0] = np.copy(fea_vector)'''
                                obj_ind[cluster_id] = obj_ind[cluster_id] + 1
                    #env_batch_slice = np.sort(env_batch_slice,axis=1)
                    env_batch[fill_index] = np.concatenate((env_batch_slice,density_array),axis=2)
                    '''here : env_batch[fill_index] = np.concatenate((env_batch_slice,density_array),axis=1)'''
                    fill_index = fill_index + 1
                    print(fill_index)
                    print('------')
                found_it = True
     #       else:
               # print(False)
    #print('max density is ',density_max)
    return env_batch[0:fill_index,],ego_batch[0:fill_index,],target_batch[0:fill_index,],(max_density_index,density_max)

def get_a_evaluation_data_batch(for_dbn):
    db = helper.connect()
    cursor = db.cursor()
    string = """SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,PT_CAMERA_COOR_ADD_OPTS.THETA,PT_CAMERA_COOR_ADD_OPTS.X_V,PT_CAMERA_COOR_ADD_OPTS.Y_V,PT_CAMERA_COOR_ADD_OPTS.CLASS,ANNOTATION.ID,OBJECT.ID,PT_CAMERA_COOR_ADD_OPTS.CLUSTER 
                    FROM PT_CAMERA_COOR,PT_CAMERA_COOR_ADD_OPTS,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID IN ('8') AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID 
                        ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"""
    cursor.execute(string)
    results = cursor.fetchall()
    
    x = map(list, list(results))
    x = list(x)
    n=np.asarray(x,dtype=np.float32)
    db.close()
    env_batch = np.zeros(shape=(BATCH_SIZE,5,6,7), dtype = np.float32)
    target_batch = np.zeros(shape=(BATCH_SIZE,2,1) , dtype = np.float32)
    ego_batch = np.zeros(shape=(BATCH_SIZE,7,1) , dtype = np.float32)
    t_start,t_end = int(n[0,4]),int(n[n.shape[0]-1,4]+1)
    fill_index = 0
    for t_id in range(t_start,t_end-2):
        c2 = n[:,4]==t_id
        c3 = n[:,4]==t_id+1
        r1 = n[c2]
        r2 = n[c3]
        object_ids_r1 = r1[:,10]
        object_ids_r2 = r2[:,10]
        '''print('looking for density :',density_2_look_4)
        if len(object_ids_r1) != density_2_look_4:
                found_it = False
                continue'''
        for o_ids in object_ids_r1:
            density_value = len(object_ids_r1)
            obj_ind = 0
            env_batch_slice = np.zeros(shape=(5,6,6), dtype = np.float32)
            '''here : env_batch_slice = np.zeros(shape=(1,6), dtype = np.float32)'''
            density_array = np.zeros(shape=(5,6,1), dtype = np.float32)
            '''here : density_array = np.zeros(shape=(1,1), dtype = np.float32)'''
            density_array.fill(len(object_ids_r1))
            for i in range(r1.shape[0]):
                if r1[i,10] == o_ids:
                    x,y = float(r1[i,2]) , float(r1[i,3])
                    if for_dbn is False:
                        fea_vector = np.reshape(np.asarray([x  , y  ,float(r1[i,6])  , float(r1[i,7])  , int(r1[i,8]),  int(r1[i,11])  , density_value ]) , newshape = (7,1))
                    else:
                        fea_vector = np.reshape(np.asarray([x  , y  , float(r1[i,5]), float(r1[i,6]) , float(r1[i,7])  , int(r1[i,8]),  int(r1[i,11]) ]) , newshape = (7,1))
                    ego_batch[fill_index] = np.copy(fea_vector)
                    for i2 in range(r2.shape[0]):
                        if r2[i2,10] == o_ids:
                            print('target entered for obj id',o_ids)
                            y_a = float(r2[i2,7]) - float(r1[i,7])
                            x_a = float(r2[i2,6]) - float(r1[i,6])
                            fea_vector = np.reshape(np.asarray([x_a,y_a]) , newshape = (2,1))
                            target_batch[fill_index] = np.copy(fea_vector)
                else:
                    #print('added env entry for object id',r1[i,10])
                    x,y = float(r1[i,2]) , float(r1[i,3])
                    cluster_id = int(r1[i,11])
                    fea_vector = np.asarray([x,y,float(r1[i,6]) , float(r1[i,7]) , int(r1[i,8])+10,  int(r1[i,11])])
                    if cluster_id < 5:
                        env_batch_slice[cluster_id,obj_ind] = np.copy(fea_vector)
                        ''' here : env_batch_slice[0] = np.copy(fea_vector)'''
                    obj_ind = obj_ind + 1
            if for_dbn:
                env_batch_slice = np.sort(env_batch_slice,axis=1)
            env_batch[fill_index] = np.concatenate((env_batch_slice,density_array),axis=2)
            '''here : env_batch[fill_index] = np.concatenate((env_batch_slice,density_array),axis=1)'''
            fill_index = fill_index + 1
            print(fill_index)
            print('------')
    print('size of eval set :',fill_index)
    return env_batch[0:fill_index,],ego_batch[0:fill_index,],target_batch[0:fill_index,],(0,0)
                      
                    
            
            
        
        
    
    
    