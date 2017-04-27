'''
Created on Dec 15, 2016

@author: atri
'''
import db_helper as helper
import numpy as np
import smooth as s
from numpy import arctan, inf
import math


class Lane_Match():
    '''
    classdocs
    '''
    

    def __init__(self,dist_cache):
        self.lanes,self.cluster_list = self.get_lanes()
        self.dist_cache = dist_cache
        '''
        Constructor
        '''
        

    def add_heading(self,list_x,list_y):
        ctr = 0
        theta_list = np.empty(shape=len(list_x))
        for i in range(len(list_x)-1):
            if (list_x[ctr+1] - list_x[ctr]) == 0 :
                slope =  inf
            else:
                slope =  (list_y[ctr+1] - list_y[ctr]) / (list_x[ctr+1] - list_x[ctr])
            theta = arctan(slope) / np.pi
            if theta < 0 :
                '''print(list_x[ctr+1], list_x[ctr] , list_y[ctr+1] , list_y[ctr] , 2 - abs(theta))'''
                theta_list[ctr] = (2 - abs(theta)) * 180
            else:
                '''print(list_x[ctr+1], list_x[ctr] , list_y[ctr+1] , list_y[ctr] , theta)'''
                theta_list[ctr] = theta * 180
            ctr = ctr + 1
        theta_list[ctr] = theta_list[ctr-1]
        return (theta_list)
    
    def get_lanes(self):
        db = helper.connect()
        cursor = db.cursor()
        string = "SELECT OBJECT.ID FROM OBJECT,ANNOTATION WHERE ANNOTATION.ID=OBJECT.PID AND ANNOTATION.FILENAME='reference'"
        cursor.execute(string)
        res = cursor.fetchone()
        lanes = []
        cluster_ids = []
        while res is not None:
            list_x,list_y = [],[]
            cursor_1 = db.cursor()
            string = "SELECT OBJECT.ID,PT_CAMERA_COOR.T, PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR_ADD_OPTS.CLUSTER FROM PT_CAMERA_COOR,OBJECT,ANNOTATION,PT_CAMERA_COOR_ADD_OPTS WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.FILENAME='reference' AND OBJECT.ID=? AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
            cursor_1.execute(string,[str(res[0])])
            res_1 = cursor_1.fetchone()
            cluster_id = None
            while res_1 is not None:
                list_x.append(float(res_1[2]))
                list_y.append(float(res_1[3]))
                if cluster_id == None:
                    cluster_id = res_1[4]
                res_1 = cursor_1.fetchone()
            
            '''smooth_list_x = s.smooth(np.asarray(list_x), window_len=3,window='flat')
            smooth_list_y = s.smooth(np.asarray(list_y), window_len=3,window='flat')'''
            z = [[e1,e2] for e1,e2 in zip(list_x,list_y)]
            a = np.asarray(z)
            smooth_list_x, smooth_list_y = s.gaussian_smoothing(a)
            theta_list = self.add_heading(smooth_list_x, smooth_list_y)
            cluster_ids.append(cluster_id)
            data_points_path = [(a,b) for a,b in zip(smooth_list_x,smooth_list_y)]
            lanes.append(data_points_path)
            res = cursor.fetchone()
        
        db.close()
        return lanes,cluster_ids   
    
    
    def heuristic_dist(self,list_a, list_b):
        '''(t1,t2) = (list_a,list_b) if len(list_a) < len(list_b) else (list_b,list_a)'''
        (t1,t2) = (list_a,list_b)
        dist = 0
        for p in t1:
            '''l = [abs(p[0] - p2s[0]) + abs(p[1] - p2s[1])  for p2s in t2]'''
            l = [math.hypot(p[0] - p2s[0], p[1] - p2s[1]) for p2s in t2]
            dist = dist + min(l)
        return dist     
        
    def match(self,data_list,tj_ts_id_tuple,partial):
        if len(data_list[0][0]) > 2:
            for i in range(len(data_list)):
                for j in range(len(data_list[i])):
                    data_list[i][j] = ( data_list[i][j][0] , data_list[i][j][1] )
            ctr = 0
        M = []
        cluster = dict()
        cluster_details = dict()
        lanes = self.lanes
        if lanes is None:
            print('ERROR')
        lane_dict = dict()
        ind = 0
        for l in lanes:
            lane_dict[ind] = l
            ind = ind + 1
        ind = 0
        for ctr in range(len(data_list)):
            proximity_dict  = dict()
            for key,value in lane_dict.items():
                if self.dist_cache is not None:
                    prev_key = str(tj_ts_id_tuple[0]) + '-' + str(tj_ts_id_tuple[1] - 1) + '-' + str(key)
                    if prev_key in self.dist_cache.keys():
                        proximity_dict[key] = self.dist_cache[prev_key] + self.heuristic_dist([data_list[ctr][-1]], value)
                        current_key = str(tj_ts_id_tuple[0]) + '-' + str(tj_ts_id_tuple[1]) + '-' + str(key)
                        self.dist_cache[current_key] = proximity_dict[key]
                    else:
                        proximity_dict[key] = self.heuristic_dist(data_list[ctr], value)
                        current_key = str(tj_ts_id_tuple[0]) + '-' + str(tj_ts_id_tuple[1]) + '-' + str(key)
                        self.dist_cache[current_key] = proximity_dict[key]
                else:        
                    proximity_dict[key] = self.heuristic_dist(data_list[ctr], value)
            lane_id_matched = min(proximity_dict, key=proximity_dict.get)
            lane_id_matched = self.cluster_list[lane_id_matched]
            if proximity_dict[lane_id_matched] == 0:
                M.append(ind)
            if lane_id_matched in cluster.keys():
                cluster[lane_id_matched].append(ind)
                cluster_details[lane_id_matched].append(data_list[ctr])
            else:
                cluster[lane_id_matched] = [ind]
                cluster_details[lane_id_matched] = [data_list[ctr]]
            ind= ind + 1
        return cluster,M,cluster_details
    
    def match2(self,data_list,agent_class,partial):
        cluster = dict()
        cluster_details = dict()
        M,M2 = [],[]
        cl = []
        ind = 0
        if partial:
            if data_list[0][0][0] < -3 and  data_list[0][0][1] > 16.7 and agent_class == '1':
                M.append(0)
            if data_list[0][0][0] > 0. and data_list[0][0][1] < 18.  and agent_class == '1':
                M.append(1)
            if data_list[0][0][0] < -5. and data_list[0][0][1] > 16.  and agent_class == '1':
                M.append(2)
            if -7 < data_list[0][0][0] < -3. and data_list[0][0][1] < 13.  and agent_class == '0':
                M.append(3)
            if data_list[0][0][0] < -5.6 and data_list[0][0][1] > 13.  and agent_class == '0':
                M.append(4)
            
            if data_list[0][-1][0] > 0. and data_list[0][-1][1] < 18  and agent_class == '1':
                M2.append(0)
            if data_list[0][-1][0] < 0. and data_list[0][-1][1] < 16  and agent_class == '1':
                M2.append(1)
            if data_list[0][-1][0] < -5. and data_list[0][-1][1] < 19  and agent_class == '1':
                M2.append(2)
            if data_list[0][-1][0] < -5.6 and data_list[0][-1][1] >13.  and agent_class == '0':
                M2.append(3)
            if data_list[0][-1][0] > -7. and data_list[0][-1][1] < 13 and agent_class == '0':
                M2.append(4)
            for i in range(6):
                if i in M and i in M2:
                    cl.append(i)
            if len(cl) > 1:
                return cl,cl,cl
            else:
                if len(M) == 0:
                    M = [0,1,2] if agent_class == '1' else [3,4] 
                    return M,M,M
                else:
                    return M,M,M
                
                
        for ctr in range(len(data_list)):
            if data_list[ctr][0][0] < -3 and data_list[ctr][0][1] > 16.7 and data_list[ctr][-1][0] > 0. and data_list[ctr][-1][1] < 18 and data_list[ctr][0][5] == '1':
                if 0 in cluster.keys():
                    cluster[0].append(ind)
                    cluster_details[0].append(data_list[ctr])
                else:
                    cluster[0] = [ind]
                    cluster_details[0] = [data_list[ctr]]
            elif data_list[ctr][0][0] > 0. and data_list[ctr][0][1] < 18. and data_list[ctr][-1][0] < 0. and data_list[ctr][-1][1] < 16 and data_list[ctr][0][5] == '1':
                if 1 in cluster.keys():
                    cluster[1].append(ind)
                    cluster_details[1].append(data_list[ctr])
                else:
                    cluster[1] = [ind]
                    cluster_details[1] = [data_list[ctr]]
            elif data_list[ctr][0][0] < -5. and data_list[ctr][0][1] > 16. and data_list[ctr][-1][0] < -5. and data_list[ctr][-1][1] < 19 and data_list[ctr][0][5] == '1':
                if 2 in cluster.keys():
                    cluster[2].append(ind)
                    cluster_details[2].append(data_list[ctr])
                else:
                    cluster[2] = [ind]
                    cluster_details[2] = [data_list[ctr]]
            elif data_list[ctr][0][0] < -3. and data_list[ctr][0][1] < 13. and data_list[ctr][-1][0] < -5.6 and data_list[ctr][-1][1] >13. and data_list[ctr][0][5] == '0':
                if 3 in cluster.keys():
                    cluster[3].append(ind)
                    cluster_details[3].append(data_list[ctr])
                else:
                    cluster[3] = [ind]
                    cluster_details[3] = [data_list[ctr]]
            elif data_list[ctr][0][0] < -5.6 and data_list[ctr][0][1] > 13. and -3 > data_list[ctr][-1][0] > -7. and data_list[ctr][-1][1] < 13 and data_list[ctr][0][5] == '0':
                if 4 in cluster.keys():
                    cluster[4].append(ind)
                    cluster_details[4].append(data_list[ctr])
                else:
                    cluster[4] = [ind]
                    cluster_details[4] = [data_list[ctr]]
            else:
                if 5 in cluster.keys():
                    cluster[5].append(ind)
                    cluster_details[5].append(data_list[ctr])
                else:
                    cluster[5] = [ind]
                    cluster_details[5] = [data_list[ctr]]
            ind= ind + 1
        return cluster,M,cluster_details
                     
        
