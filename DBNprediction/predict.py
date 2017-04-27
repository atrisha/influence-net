'''
Created on Dec 19, 2016

@author: atri
'''

import numpy as np
import pickle
from lane_match import *
from collections import OrderedDict
from scipy.spatial import distance


traj_cache = dict()
tail_traj_cache = dict()
def order_by(p):
    return p[0]

def has_key(gt,k,t):
    if t-1 in gt.keys():
        if k in gt[t-1].keys():
            return True
    return False

def heuristic_dist(list_a, list_b):
    '''(t1,t2) = (list_a,list_b) if len(list_a) < len(list_b) else (list_b,list_a)'''
    (t1,t2) = (list_a,list_b)
    dist = 0
    for p in t1:
        '''l = [abs(p[0] - p2s[0]) + abs(p[1] - p2s[1])  for p2s in t2]'''
        l = [math.hypot(p[0] - p2s[0], p[1] - p2s[1]) for p2s in t2]
        dist = dist + min(l)
    return dist

def get_theta_from_sine(disp_x,disp_y,sin_new_theta):
    val = np.degrees(np.arcsin(sin_new_theta))
    if disp_x > 0 and disp_y > 0:
        val = val
    elif disp_x < 0 and disp_y > 0:
        val = 180 - abs(val)
    elif disp_x < 0 and disp_y < 0:
        val =180 + abs(val)
    elif disp_x > 0 and disp_y < 0:
        val = 360 + val
    elif disp_x < 0 and disp_y == 0:
        val = 180
    elif disp_x == 0 and disp_y < 0:
        val = 270
    val = np.deg2rad(val)
    return val

def modulo_add(a,b):
    r = a + b
    if r<-1 :
        if r<-2:
            return -2 - r
        else:
            return (-1 + (-1 - r))
    elif r > 1 :
        if r > 2:
            return 2 -r
        else:
            return (1 - (r-1))
    else:
        return r

def print_summary(results_dict_target,results_dict_predicted):
    summary_dict_vehicles = dict()
    summary_dict_pedestrians = dict()
    for t in results_dict_target.keys():
        for window in range(1,91):
            if t in results_dict_predicted.keys():
                if t+window in results_dict_predicted[t].keys():
                    for agent in results_dict_predicted[t][t+window].keys():
                        agent_id,_,agent_class = agent.partition('#')
                        x_p,y_p = results_dict_predicted[t][t+window][agent][0] , results_dict_predicted[t][t+window][agent][1]
                        if t+window in results_dict_target.keys():
                            if agent in results_dict_target[t+window].keys():
                                x,y = results_dict_target[t+window][agent][0] , results_dict_target[t+window][agent][1]
                                delta = math.sqrt((x - x_p)**2 + (y - y_p)**2)
                                if agent_class == '1.0':
                                    if window in summary_dict_vehicles.keys() :
                                        summary_dict_vehicles[window].append(delta)
                                    else:
                                        summary_dict_vehicles[window] = [delta]
                                else:
                                    if window in summary_dict_pedestrians.keys() :
                                        summary_dict_pedestrians[window].append(delta)
                                    else:
                                        summary_dict_pedestrians[window] = [delta]
    print('-------------------------vehicles---------------------------')
    for k,v in summary_dict_vehicles.items():
        print(k,np.mean(v))
    print('-------------------------pedestrians---------------------------')
    for k,v in summary_dict_pedestrians.items():
        print(k,np.mean(v))
    pickle.dump(summary_dict_vehicles, open( "save_result_summary_vehicles_dbn.p", "wb" ))
    pickle.dump(summary_dict_pedestrians, open( "save_result_summary_pedestrians_dbn.p", "wb" ))
    
            

def get_projected_points(cluster_details,new_cluster_id,disp,trail,t_id,agent_id,window_id,head,tail):
    '''
    3 step process:
    1. extrals trajectories from cluster_details[cluster_id]
    2. from all the trajectories find the closest match.
    3. Find arc length point in the matched trajectory till disp length
    
    '''
    
    global traj_cache
    dist = inf
    matched_ind = -1
    matched_trajs = cluster_details[new_cluster_id]
    'matched traj id'
    for ind,trajs in enumerate(matched_trajs):
        k = str(t_id) + '-' + str(agent_id) + '-' + str(window_id-1) + '-' + str(ind)
        if k in traj_cache:
            #print('hit',k)
            curr_dist = traj_cache[k] + heuristic_dist([trail[-1]], trajs)
            new_k = str(t_id) + '-' + str(agent_id) + '-' + str(window_id) + '-' + str(ind)
            traj_cache[new_k] = curr_dist
        else:
            t_k = str(t_id-1) + '-' + str(agent_id) + '-' + str(ind)
            if t_k in tail_traj_cache.keys():
                #print('hit',k)
                curr_dist = tail_traj_cache[t_k] + heuristic_dist(head, trajs)
                new_t_k = str(t_id) + '-' + str(agent_id) + '-' + str(ind)
                tail_traj_cache[new_t_k] = curr_dist
                new_k = str(t_id) + '-' + str(agent_id) + '-' + str(window_id) + '-' + str(ind)
                traj_cache[new_k] = curr_dist
            else:
                #print('created',k)
                curr_dist = heuristic_dist(tail, trajs) + heuristic_dist(head, trajs) 
                new_k = str(t_id) + '-' + str(agent_id) + '-' + str(window_id) + '-' + str(ind)
                traj_cache[new_k] = curr_dist
                new_t_k = str(t_id) + '-' + str(agent_id) + '-' + str(ind)
                tail_traj_cache[new_t_k] = curr_dist
                
        if curr_dist < dist:
            matched_ind = ind
            dist = curr_dist
    predicted_path = matched_trajs[matched_ind]
    
    dist = inf
    closest_ind = -1
    last_pt = trail[-1]
    for ind,pts in enumerate(predicted_path):
        curr_dist = math.hypot(pts[0] - last_pt[0], pts[1] - last_pt[1])
        if curr_dist < dist:
            closest_ind = ind
            dist = curr_dist
    closest_pts = predicted_path[closest_ind]
    path_remaining = predicted_path[closest_ind:]    
    sum = 0
    projected_points = None
    if len(path_remaining) == 1:
        return None
    if ( path_remaining[1][0] - path_remaining[0][0] ) == 0:
        return np.arctan(np.inf)
    else:
        angle = arctan((path_remaining[1][1] - path_remaining[0][1]) / (path_remaining[1][0] - path_remaining[0][0]))
    '''for i in range(1,len(path_remaining)):
        sum = sum + math.hypot(path_remaining[i][0] - path_remaining[i-1][0], path_remaining[i][1] - path_remaining[i-1][1])
        if sum >= disp:
            projected_points = (path_remaining[i-1])
            break'''
    return angle

def euclid_dist(x2,x1,y2,y1):
    return math.sqrt((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2)

def get_projected_point_2(dist,path,origin):
    ''' add origin point to the closet point in the path direction and trim path'''
    min_dist = np.inf
    future_path_index = None
    for i in range(len(path)):
        curr_dist =  euclid_dist(path[i][0], origin[0], path[i][1], origin[1])
        if curr_dist < min_dist:
            min_dist = curr_dist
            future_path_index = i
    if path == None or future_path_index == None or min_dist > 20:
        print(min_dist)
        return None
    #print('future_path_index', future_path_index)
    '''future_path_index = 0
    for i in range(len(path) -1):
        if (path[i][0] <= origin[0] <= path[i+1][0]) and (path[i][1] <= origin[1] <= path[i+1][1]):
            future_path_index = i+1
            break'''
    path = path[future_path_index:]           
    dist_sum = euclid_dist(path[0][0], origin[0], path[0][1], origin[1])
    last_sum = dist_sum
    for i in range(len(path) -1):
        if dist_sum >= dist:
            v_vector = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if v_vector[0] == 0 and v_vector[1] == 0:
                continue
            overshoot_dist = dist - last_sum
            unit_vector = ( ( v_vector[0] ) / euclid_dist(path[i+1][0], path[i][0], path[i+1][1], path[i][1]) ,
                            ( v_vector[1] ) / euclid_dist(path[i+1][0], path[i][0], path[i+1][1], path[i][1]) )
            new_x, new_y = (path[i][0] + unit_vector[0] * overshoot_dist, path[i][1] + unit_vector[1] * overshoot_dist)
            return (new_x,new_y)
        else:
            last_sum = dist_sum
            dist_sum = dist_sum + euclid_dist(path[i+1][0], path[i][0], path[i+1][1], path[i][1])
            
def cumulative_dist(v,tail):
    '''length = len(tail)
    dist = 0
    if length > len(v):
        return -1
    for i in range(length):
        dist = dist + euclid_dist(v[i][0],tail[i][0],v[i][1],tail[i][1])'''
    ''' find index1 in v closest to tail[0] find index2 in v closest to tail[1]'''
    ''' from index1 to index2 in v sum min dist between v[index] to tail'''
    ''' divide sum by index2 - index1 to get per point distance'''
    dist_from_tail_0 = [euclid_dist(v[i][0],tail[0][0],v[i][1],tail[0][1]) for i,d in enumerate(v)]
    index_1 = np.argmin(dist_from_tail_0)
    dist_from_tail_n = [euclid_dist(v[i][0],tail[-1][0],v[i][1],tail[-1][1]) for i,d in enumerate(v)]
    index_2 = np.argmin(dist_from_tail_n)
    dist_sum = 0
    for i in range(min(index_1,index_2) , max(index_1,index_2)+1):
        dist_from_tail = [euclid_dist(v[i][0],t[0],v[i][1],t[1]) for t in tail]
        dist_sum = dist_sum + min(dist_from_tail)
    dist = dist_sum / (abs(index_2 - index_1) + 1)
    return dist
    
def get_path(agent_id,t,tail,paths_array):
    length = len(tail)
    min_dist = np.inf
    path = []
    for k,v in paths_array.items():
        dist = cumulative_dist(v, tail)
        if dist == -1 :
            continue
        elif dist < min_dist:
            min_k = k
            path = v[1:]
            min_dist = dist
    #head_dist = euclid_dist(path[0][0], tail[-1][0], path[0][1], tail[-1][1])
    #print('path',path,'head dist',head_dist)
    return path
    ''' len(path) > 2:
        x,y=[],[]
        for i in path:
            x.append(float(i[0]))
            y.append(float(i[1]))
        z = np.polyfit(x, y, 3)
        return z
    else:
        return path'''        
        
def predict():
    fps = 30
    db = helper.connect()
    cursor = db.cursor()
    string = "SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,PT_CAMERA_COOR_ADD_OPTS.THETA,PT_CAMERA_COOR_ADD_OPTS.X_V,PT_CAMERA_COOR_ADD_OPTS.Y_V,PT_CAMERA_COOR_ADD_OPTS.CLASS FROM PT_CAMERA_COOR,PT_CAMERA_COOR_ADD_OPTS,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID IN ('8') AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
    cursor.execute(string)
    results = cursor.fetchall()
    
    
    mean_functions = pickle.load( open( "save_mean_functions_8.p", "rb" ) )
    model_params = pickle.load( open( "save_model_params_8.p", "rb" ) )
    
    
    print(mean_functions.shape,model_params.shape)
    
    
        
    
    x = map(list, list(results))
    x = list(x)
    n=np.asarray(x,dtype=np.float32)
    lm = Lane_Match(dict())
    ground_truth = OrderedDict()
    predicted_path = OrderedDict()
    db.close()
    results_dict_target = dict()
    results_dict_predicted = dict()
    #for t in range(int(n[0,4]),int(n[n.shape[0]-1,4]+1)):
    for t in range(int(n[0,4]) , int(n[0,4]) + 800):
        w1 = np.where((n[:,4] == t))
        density = len(w1[0])
        prediction_input = []
        agent_list = []
        for objs in n[w1[0],:]:
            agent_list.append([objs[2],objs[3],objs[5],objs[6],objs[7],objs[8],objs[1]])
            #agent_list.append(x,y,theta,x_v,y_v,class,id,cluster_id)
        agent_list = sorted(agent_list, key = order_by)
        '''for agents in agent_list:
            prediction_input = prediction_input + agents
        prediction_input = np.asarray(prediction_input)
        '''
        '''print('will predict with density:',density,':',prediction_input)
        prediction_ouput = []
        for ag in range(density):
            for x_y in range(3):
                predictor = mean_functions[density,(3*ag)+x_y]
                rf_model = pickle.loads(predictor)
                res = rf_model.predict(prediction_input)
                prediction_ouput = prediction_ouput + res
        '''
        agent_cluster_map = dict()
        agent_path_map,head_dist = dict(),dict()
        ground_truth[t] = OrderedDict()
        for agents in agent_list:
            agent_id = int(agents[6])
            if t == 106:
                deb = True
            if has_key(ground_truth, agent_id,t):
                ground_truth[t][agent_id] = ground_truth[t-1][agent_id] + [(agents[0],agents[1])]
                predicted_path[str(agent_id)+'-'+str(t)] = [(agents[0],agents[1])]
            else:
                ground_truth[t][agent_id] = [(agents[0],agents[1])]
                predicted_path[str(agent_id)+'-'+str(t)] = [(agents[0],agents[1])]
            agent_class = str(agents[5])
            if t in results_dict_target.keys():
                results_dict_target[t][str(agent_id) +'#'+ agent_class] = (agents[0],agents[1])
            else:
                results_dict_target[t] = dict()
                results_dict_target[t][str(agent_id) +'#'+ agent_class] = (agents[0],agents[1])

            cluster,M,dummy = lm.match2([ground_truth[t][agent_id]],str(int(float(agent_class))),True)
            '''if len(cluster) == 0 :
                cluster,M,dummy = lm.match([ground_truth[t][agent_id]],(agent_id,t),True)
                cluster = list(cluster.keys())
            #cluster_id = list(cluster.keys())[0]'''
            cluster_id = cluster
            tail = list(ground_truth[t][agent_id])
            '''
            get trajectories of agents with cluster_id.
            find the trajectory closest to tail
            agent_path_map[agent_id] = [(x,y) tuples of closest trajectory]
            '''
            #print('agent',agent_id,'in clusters',cluster_id,'at t',t)
            path_map = dict()
            db = helper.connect()
            cursor = db.cursor()
            string = "SELECT PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y FROM PT_CAMERA_COOR,PT_CAMERA_COOR_ADD_OPTS WHERE PT_CAMERA_COOR_ADD_OPTS.CLUSTER IN ("
            for num_cl in range(len(cluster_id)-1):
                string = string + "?,"
            string = string + "? ) AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID AND PT_CAMERA_COOR.PID NOT IN (?) ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
            q_list = cluster_id + [str(agent_id)]
            #print(string,q_list)
            cursor.execute(string,q_list)
            res_1 = cursor.fetchone()
            while res_1 is not None:
                id = res_1[0]
                if id in path_map.keys():
                    path_map[id] = path_map[id] + [(float(res_1[1]),float(res_1[2]))] 
                else:
                    path_map[id] = [(float(res_1[1]),float(res_1[2]))]
                res_1 = cursor.fetchone()
            
            agent_path_map[agent_id] = get_path(agent_id,t,tail,path_map)
            
            
            db.close()
            ''' comment the following line for dynamically changing cluster id with prediction'''
            #agent_cluster_map[agent_id] = cluster_id
            prediction_input = prediction_input + [agents[0],agents[1],agents[2],agents[3],agents[4],agents[5],cluster_id[0]]
            
        prediction_input = np.asarray(prediction_input).reshape(1,-1)
        
        
        
        prediction_at_current_ts = OrderedDict()
        for agents in agent_list:
            prediction_at_current_ts[int(agents[6])] = []
        for window in range(90):
            cnt = 0
            for agents in agent_list:
                agent_id = int(agents[6])
                agent_class = str(agents[5])
                head = prediction_at_current_ts[agent_id]
                tail = list(ground_truth[t][int(agents[6])])
                if prediction_at_current_ts[agent_id] is None:
                    trail = list(ground_truth[t][int(agents[6])])
                else:
                    trail = list(ground_truth[t][int(agents[6])]) + prediction_at_current_ts[agent_id]
                
                ''' uncomment following 2 lines for dynamically changing cluster id with prediction'''
                cluster,M,dummy = lm.match2([trail],str(int(float(agent_class))),True)
                '''if len(cluster) == 0 :
                    cluster,M,dummy = lm.match([ground_truth[t][agent_id]],(agent_id,t),True)
                    cluster = list(cluster.keys())'''
                new_cluster_id = cluster[0]
                #new_cluster_id = agent_cluster_map[agent_id]
                
                predictor_x_a = mean_functions[density,cnt*3,0]
                predictor_y_a = mean_functions[density,cnt*3 + 1,0]
                predictor_omega = mean_functions[density,cnt*3 + 2,0]
                prediction_input = prediction_input.astype(np.float32)
                x_a = np.random.normal(loc = predictor_x_a.predict(prediction_input) , scale = np.sqrt(model_params[density,cnt*3,1]))
                y_a = np.random.normal(loc = predictor_y_a.predict(prediction_input) , scale = np.sqrt(model_params[density,cnt*3 + 1,1]))
                '''
                omega = sin(new_theta) - sin(old_theta) ;
                new_theta = arcsin(omega + sin(old_theta)_
                '''
                
                
                disp_x = (prediction_input[0,cnt*7 + 3] * (1/fps)) + (0.5 * float(x_a) * math.pow((1/fps), 2))
                disp_y = (prediction_input[0,cnt*7 + 4] * (1/fps)) + (0.5 * float(y_a) * math.pow((1/fps), 2))
                
                disp = math.pow((math.pow(disp_x,2) + math.pow(disp_y, 2)),0.5)
                
                ''' window , agent_id , t '''
                #got_theta = get_projected_points(cluster_details,new_cluster_id,disp,trail,t,agent_id,window,head,tail)
                got_theta = None
                if got_theta is not None:
                    new_theta = got_theta
                else:
                    omega = np.random.normal(loc = predictor_omega.predict(prediction_input) , scale = np.sqrt(model_params[density,cnt*3 + 2,1]))
                    # Draw from distribution
                    sin_new_theta = modulo_add(omega , np.sin(prediction_input[0,cnt*7 + 2]))
                    new_theta = get_theta_from_sine(disp_x,disp_y,sin_new_theta)
                    #print(omega + np.sin(prediction_input[0,cnt*7 + 2]), new_theta)
                
                new_theta_deg = np.rad2deg(new_theta)
                
                #new_x = prediction_input[0,cnt*7] + disp * math.cos(new_theta)
                #new_y = prediction_input[0,cnt*7 + 1] + disp * math.sin(new_theta)
                
                '''
                find next x,y in the direction of agent_path_map[agent_id]
                '''
                path = agent_path_map[agent_id]
                new_points = get_projected_point_2(disp, path , (prediction_input[0,cnt*7] , prediction_input[0,cnt*7 + 1]))
                if new_points == None:
                    #print('no path found at t',t,'w',window,'for agent',agent_id)
                    #new_x = prediction_input[0,cnt*7] + disp * math.cos(new_theta)
                    #new_y = prediction_input[0,cnt*7 + 1] + disp * math.sin(new_theta)
                    continue
                else:
                    #print('new points',new_points, len(tail))
                    new_x,new_y = new_points[0],new_points[1]
                
                trail.append((new_x,new_y))
                new_x_v = prediction_input[0,cnt*7 + 3] + float(x_a) * (1/fps)
                new_y_v = prediction_input[0,cnt*7 + 4] + float(y_a) * (1/fps)
                
                prediction_at_current_ts[int(agents[6])].append((new_x,new_y))
                
                if t in results_dict_predicted.keys():
                    if t+window+1 in results_dict_predicted[t].keys():
                        results_dict_predicted[t][t+window+1][str(agent_id) +'#'+ agent_class] = (new_x,new_y)
                    else:
                        results_dict_predicted[t][t+window+1] = dict()
                        results_dict_predicted[t][t+window+1][str(agent_id) +'#'+ agent_class] = (new_x,new_y)
                else:
                    results_dict_predicted[t] = dict()
                    results_dict_predicted[t][t+window+1] = dict()
                    results_dict_predicted[t][t+window+1][str(agent_id) +'#'+ agent_class] = (new_x,new_y)
                
                
                prediction_input[0,cnt*7] = new_x
                prediction_input[0,cnt*7 + 1] = new_y
                prediction_input[0,cnt*7 + 2] = new_theta
                prediction_input[0,cnt*7 + 3] = new_x_v
                prediction_input[0,cnt*7 + 4] = new_y_v
                prediction_input[0,cnt*7 + 6] = new_cluster_id
                cnt = cnt + 1
        
        for agents in agent_list:
            agent_id = int(agents[6])
            predicted_path[str(agent_id)+'-'+str(t)] = predicted_path[str(agent_id)+'-'+str(t)] + \
                                                    prediction_at_current_ts[agent_id] 
                
        print('-------- : ',t)
    pickle.dump(ground_truth, open( "save_ground_truth.p", "wb" ))
    pickle.dump(predicted_path, open( "save_predicted_path.p", "wb" ))
    
    print_summary(results_dict_target,results_dict_predicted)
    
if __name__ == '__main__':
    predict()    
    
