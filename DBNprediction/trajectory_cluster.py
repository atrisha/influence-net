'''
Created on Nov 28, 2016

@author: atri
'''


import db_helper as helper
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import smooth as s
from numpy import arctan, inf
import sys
from lane_match import *
import math
import pickle

fps = 30

def add_heading(list_x,list_y):
    ctr = 0
    theta_list = []
    for i in range(len(list_x)-1):
        X = [list_x[j] for j in (i,i+1)]
        Y = [list_y[j] for j in (i,i+1)]
        theta = np.arctan2(Y[1] - Y[0], X[1] - X[0])
        theta_list.append(theta)
    theta_list.append(theta_list[-1])
    theta_list = np.asarray(np.degrees(theta_list) % 360)
    return theta_list


def print_stats(some_list,id):
    sl_as_np = np.asarray(some_list)
    '''plt.hist(some_list,bins=100)
    plt.show()'''
    print(id,':', ' min:',np.min(sl_as_np), ' med:',np.median(sl_as_np),' max:',np.max(sl_as_np),' 10-q:',np.percentile(sl_as_np,10),' 25-q:',np.percentile(sl_as_np,25),' 75-q:',np.percentile(sl_as_np,75),' 90-q:',np.percentile(sl_as_np,90),' 95-q:',np.percentile(sl_as_np,95))
    

def form_input():
    db = helper.connect()
    cursor = db.cursor()
    string = "SELECT (SELECT COUNT(*) FROM OBJECT), (SELECT COUNT(*) FROM ANNOTATION);"
    cursor.execute(string)
    res = cursor.fetchone()
    count,count_scenarios = res[0],res[1]
    max_x,min_x,max_y,min_y = float('-inf'),float('inf'),float('-inf'),float('inf')
    data_path,data_trajectory = dict(),dict()
    particle_ids = []
    ref_indexes = []
    for i in range(1,count+1) :
        cursor = db.cursor()
        string = "SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,OBJECT.NAME,ANNOTATION.ID,OBJECT.STARTFRAME,OBJECT.ENDFRAME FROM PT_CAMERA_COOR,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND OBJECT.ID=? AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID NOT IN ('10') ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
        cursor.execute(string,[i])
        res_1 = cursor.fetchone()
        if res_1 is None:
            continue
        h = int(res_1[6])
        
        if h == 3:
            ref_indexes.append(i) 
            
        last_x,last_y = None,None
        list_x,list_y,list_type,list_x_v,list_y_v=[],[],[],[],[]
        traj_length = 0
        while res_1 is not None:
            t_id = res_1[0]
            ''' since the trajectories are ordered now, and will be merged again in order again, there is no need to keep track of the indices explicitly for particle ids'''
            particle_ids.append(t_id)
            traj_length = traj_length + 1
            if max_x < float(res_1[2]):
                max_x = float(res_1[2])
            if min_x > float(res_1[2]):
                min_x = float(res_1[2])
            if max_y < float(res_1[3]):
                max_y = float(res_1[3])
            if min_y > float(res_1[3]):
                min_y = float(res_1[3])
            if last_x is not None and last_y is not None :
                velocity_x, velocity_y = (float(res_1[2]) - last_x) * fps , (float(res_1[3]) - last_y) * fps
                list_x_v.append(velocity_x)
                list_y_v.append(velocity_y)
            last_x,last_y = float(res_1[2]) , float(res_1[3])
            list_x.append(float(res_1[2]))
            list_y.append(float(res_1[3]))
            type_str = str(res_1[5][:3])
            if 'veh' == type_str :
                list_type.append('1')
            elif 'ped' == type_str :
                list_type.append('0')
            else :
                print(False)
            
            res_1 = cursor.fetchone()
        list_x_v.insert(0, list_x_v[0])
        list_y_v.insert(0, list_y_v[0])
        list_x_a = list(map(lambda v1, v2 : (v2 - v1), list_x_v , list_x_v[1:] + [list_x_v[-1]]))
        list_y_a = list(map(lambda v1, v2 : (v2 - v1), list_y_v , list_y_v[1:] + [list_y_v[-1]]))
        '''smooth_list_x = s.smooth(np.asarray(list_x), window_len=10,window='flat')
        smooth_list_y = s.smooth(np.asarray(list_y), window_len=10,window='flat')
        smooth_list_x = s.smooth(smooth_list_x)
        smooth_list_y = s.smooth(smooth_list_y)'''
        ''''z = [[e1,e2] for e1,e2 in zip(list_x,list_y)]
        a = np.asarray(z)
        smooth_list_x, smooth_list_y = s.gaussian_smoothing(a)'''
        smooth_list_x, smooth_list_y = list_x,list_y
        #print('size-x',len(list_x),len(smooth_list_x))
        #print('size-y',len(list_y),len(smooth_list_y))
        theta_list = add_heading(smooth_list_x, smooth_list_y)
        theta_list = np.deg2rad(theta_list)
        if not np.all(np.isfinite(np.asarray(theta_list))) or np.any(np.isnan(np.asarray(theta_list))):
            print(theta_list)
        ''' theta goes in as radians. omega goes in as sin(theta2) - sin(theta1)'''
        omega_list = list(map(lambda v1, v2 : (v2 - v1), list(np.sin(theta_list)) , list(np.sin(theta_list))[1:] + [list(np.sin(theta_list))[-1]]))    
        '''print_stats(omega_list,i)'''
        data_points_path = [(a,b,c) for a,b,c in zip(smooth_list_x,smooth_list_y,theta_list)]
        data_points_trajectory = [(a,b,c,d,e,f,g1,h1,i1) for a,b,c,d,e,f,g1,h1,i1 in zip(smooth_list_x,smooth_list_y,theta_list,list_x_v,list_y_v,list_type,list_x_a,list_y_a,omega_list)]
        if h in data_path:
            data_path[h].append(data_points_path)
            data_trajectory[h].append(data_points_trajectory)
        else:
            data_path[h] = [data_points_path]
            data_trajectory[h] = [data_points_trajectory]
        
    db.close()
    data_path_norm = data_path.copy()
    
    ''' Normalization'''
    '''
    for k in data_path_norm:
        for i in range(len(data_path_norm[k])):
            for j in range(len(data_path_norm[k][i])):
                data_path_norm[k][i][j] = ((data_path_norm[k][i][j][0] - min_x ) / (max_x - min_x) , (data_path_norm[k][i][j][1] - min_y ) / (max_y - min_y) , (data_path_norm[k][i][j][2]) / 360)
    '''            
    
    
    return data_path,data_trajectory,particle_ids

def on_plot_hover(event):
    for curve in plt.get_lines():
        if curve.contains(event)[0]:
            print("over", curve.get_gid())
            
    

def get_cl_id(f,C):
    for k,v in C.items(): 
        if f in v: 
            return k            



def cluster():
    '''
    This clusters/lane matches trajectory data contained in MySQL DB (parameters are in config/connection.ini).
    Refer to the project : bitbucket.org/a9sarkar/traffic-data/ for the structure of the DB.
    Essentially, this is the training data to build the BN model.
    
    Returns a cluster of trajectory data mapped to scenario density (i.e. number of participants in the scene/frame)
    '''
    print('clustering start')
    data,data_trajectory,particle_ids = form_input()
    'this maintains a map between the scenario_id and the trajectory indexes in the merged data'
    sampling_tuple = (13,10)
    merged_data_norm = []
    ''' assumption : keys will be in order'''
    for key,value in data_trajectory.items():
        merged_data_norm = merged_data_norm + value
    
    lm = Lane_Match(None)
    #C,M,cluster_details = lm.match(merged_data_norm,None)
    C,M,cluster_details = lm.match2(merged_data_norm,None,False)
    merged_data = []
    for key,value in data.items():
        merged_data = merged_data + value
        
    ''' assumption : merging will preserve the order'''
    merged_data_trajectory = []
    tj_index = 0
    for key,value in data_trajectory.items():
        for trajectories in value:
            cl_id = get_cl_id(tj_index, C)
            for particle in trajectories:
                particle_l = list(particle) + [cl_id]
                merged_data_trajectory.append(tuple(particle_l))
            tj_index = tj_index + 1
    
    ''' First, remove the heading vector'''
    for i in range(len(merged_data)):
        for j in range(len(merged_data[i])):
            merged_data[i][j] = ( merged_data[i][j][0] , merged_data[i][j][1] )
    
    color_map = ['b','lawngreen','coral','violet','k','white']
    fig = plt.figure()
    plot = fig.add_subplot(111)
    
    for i in range(len(merged_data)):
        for k,v in C.items():
            if i in v:
                plot.plot(*zip(*merged_data[i]), color_map[k], gid = k)
    def on_plot_hover(event):
        for curve in plot.get_lines():
            if curve.contains(event)[0]:
                print("over ", curve.get_gid())
                
    fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    plt.show()
    
    return merged_data_trajectory,sampling_tuple,M,C,particle_ids
    

    
if __name__ == '__main__':
    cluster()   
           
    