'''
Created on Feb 1, 2017

@author: atrisha
'''

import tensorflow as tf
import numpy as np
import time
import sys
import flags
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import env_summary
from lane_match import *
from collections import OrderedDict
import pickle

GRID_SIZE_ROWS = flags.GRID_SIZE_ROWS
GRID_SIZE_COLS = flags.GRID_SIZE_COLS
NUM_INPUT_CHANNELS = flags.NUM_INPUT_CHANNELS
NUM_ENV_CHANNELS = flags.NUM_ENV_CHANNELS
NUM_OUTPUT_CHANNELS = flags.NUM_OUTPUT_CHANNELS
BATCH_SIZE = flags.BATCH_SIZE
DECAY_STEP = flags.DECAY_STEP
NUM_EPOCHS = flags.NUM_EPOCHS
SEED = flags.SEED
EVAL_BATCH_SIZE = flags.EVAL_BATCH_SIZE
EVAL_FREQUENCY = flags.EVAL_FREQUENCY

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
    pickle.dump(summary_dict_vehicles, open( "save_result_summary_vehicles_nn_8.p", "wb" ))
    pickle.dump(summary_dict_pedestrians, open( "save_result_summary_pedestrians_nn_8.p", "wb" ))
    

def order_by(p):
    return p[0]

def has_key(gt,k,t):
    if t-1 in gt.keys():
        if k in gt[t-1].keys():
            return True
    return False

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def euclid_dist(x2,x1,y2,y1):
    return math.sqrt((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2)

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


def get_projected_point_2(dist,path,origin):
    ''' add origin point to the closet point in the path direction and trim path'''
    min_dist = np.inf
    future_path_index = None
    for i in range(len(path)):
        curr_dist =  euclid_dist(path[i][0], origin[0], path[i][1], origin[1])
        if curr_dist < min_dist:
            min_dist = curr_dist
            future_path_index = i
    if path == None or future_path_index == None or min_dist > 2:
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
            
def get_real_frames():
    real_frame_dict = dict()
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
    t_start,t_end = int(n[0,4]),int(n[n.shape[0]-1,4]+1)
    for t_id in range(t_start,t_end-2):
        if t_id == 560:
            deb = 5
        c2 = n[:,4]==t_id
        c3 = n[:,4]==t_id+1
        r1 = n[c2]
        r2 = n[c3]
        object_ids_r1 = r1[:,10]
        object_ids_r2 = r2[:,10]
        for o_ids in object_ids_r1:
            density_value = len(object_ids_r1)
            obj_ind = 0
            env_batch_slice = np.zeros(shape=(5,6,6), dtype = np.float32)
            for i in range(r1.shape[0]):
                if r1[i,10] == o_ids:
                    x,y = float(r1[i,2]) , float(r1[i,3])
                    fea_vector = np.reshape(np.asarray([x  , y  ,float(r1[i,6])  , float(r1[i,7])  , int(r1[i,8]),  int(r1[i,11])  , density_value ]) , newshape = (7,1))
                    if t_id in real_frame_dict.keys():
                        real_frame_dict[t_id][o_ids] = {'ego' : fea_vector}
                    else:
                        real_frame_dict[t_id] = dict()
                        real_frame_dict[t_id][o_ids] = dict()
                        real_frame_dict[t_id][o_ids]['ego'] = np.copy(fea_vector)
                    for i2 in range(r2.shape[0]):
                        if r2[i2,10] == o_ids:
                            print('target entered for obj id',o_ids)
                            y_a = float(r2[i2,7]) - float(r1[i,7])
                            x_a = float(r2[i2,6]) - float(r1[i,6])
                            fea_vector = np.reshape(np.asarray([x_a,y_a]) , newshape = (2,1))
                            real_frame_dict[t_id][o_ids]['target'] = np.copy(fea_vector)
                else:
                    #print('added env entry for object id',r1[i,10])
                    x,y = float(r1[i,2]) , float(r1[i,3])
                    cluster_id = int(r1[i,11])
                    fea_vector = np.asarray([x,y,float(r1[i,6]) , float(r1[i,7]) , int(r1[i,8])+10,  int(r1[i,11])])
                    if cluster_id < 5:
                        env_batch_slice[cluster_id,obj_ind] = np.copy(fea_vector)
                        ''' here : env_batch_slice[0] = np.copy(fea_vector)'''
                    obj_ind = obj_ind + 1
            env_vector = env_summary.encode_env(env_batch_slice)
            real_frame_dict[t_id][o_ids]['env'] = np.copy(env_vector)
            
    
    return real_frame_dict


def main(_):
    num_epochs = NUM_EPOCHS
    '''here : train_data_env_node = tf.placeholder(tf.float32, [GRID_SIZE_ROWS,GRID_SIZE_COLS,NUM_INPUT_CHANNELS])'''
    data_env_node = tf.placeholder(tf.float32, [1,NUM_ENV_CHANNELS])
    data_ego_node = tf.placeholder(tf.float32, [NUM_INPUT_CHANNELS,1])
    target_node = tf.placeholder(tf.float32, [NUM_OUTPUT_CHANNELS,1])
    
    def model(data_env,data_ego):
    #def model(data_ego):    
        #data_env = tf.nn.l2_normalize(data_env, dim=[0,1])
        
        input_layer = tf.reshape(data_env, [NUM_ENV_CHANNELS, 1])
        
        with tf.variable_scope('hidden1') as scope:
            #weight_layer_1 = tf.Variable(tf.random_gamma(shape = [NUM_ENV_CHANNELS,10], alpha = 1.0, beta = 0.5))
            weight_layer_1 = tf.Variable(tf.random_normal(shape = [NUM_ENV_CHANNELS,10]))
            variable_summaries(weight_layer_1)
            biases_layer_1 = tf.Variable(tf.fill([10], 0.1))
            hidden_layer = tf.nn.tanh(tf.matmul(tf.transpose(input_layer), weight_layer_1) + biases_layer_1, name=scope.name)
        
        with tf.variable_scope('output1') as scope:
            weight_layer_2 = tf.Variable(tf.random_normal([10,3]))
            biases_layer_1 = tf.Variable(tf.zeros([1,3], dtype=tf.float32))
            output_1 = tf.transpose(tf.nn.tanh(tf.matmul(hidden_layer, weight_layer_2) + biases_layer_1, name=scope.name)) # 5 X 1
            tf.summary.histogram('activations', output_1)
            
        ''' data_ego should be in shape 7 X 1'''
            
        input_2 = tf.concat(0,[data_ego,output_1]) # 9 X 1
        
        
        with tf.variable_scope('hidden2') as scope:
            weight_layer_3 = tf.Variable(tf.random_normal([10, 20]))
            biases_layer_3 = tf.Variable(tf.zeros([20], dtype=tf.float32))
            hidden_layer_3 = tf.nn.tanh(tf.matmul(tf.transpose(input_2), weight_layer_3) + biases_layer_3, name=scope.name)
            
        with tf.variable_scope('hidden3') as scope:
            weight_layer_5 = tf.Variable(tf.random_normal([20, 10]))
            biases_layer_5 = tf.Variable(tf.zeros([10], dtype=tf.float32))
            hidden_layer_5 = tf.nn.tanh(tf.matmul(hidden_layer_3, weight_layer_5) + biases_layer_5, name=scope.name)
            
        with tf.variable_scope('output2') as scope:    
            weight_layer_4 = tf.Variable(tf.random_normal([10,2]))
            biases_layer_4 = tf.Variable(tf.zeros([1,2], dtype=tf.float32))
            output_2 = tf.transpose(tf.matmul(hidden_layer_5, weight_layer_4) + biases_layer_4, name=scope.name) # 2 X 1
            
        return output_2
    
    output_vector = model(data_env_node, data_ego_node)
    #eval_prediction = model(eval_data_env_node,eval_data_ego_node)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    start_time = time.time()
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter('./train')
        print('Initialized!')
        
        saver.restore(sess, "./model.ckpt")
        print("Model restored.")
        eval_loss = []
        
        fps = 30
        db = helper.connect()
        cursor = db.cursor()
        string = "SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,PT_CAMERA_COOR_ADD_OPTS.THETA,PT_CAMERA_COOR_ADD_OPTS.X_V,PT_CAMERA_COOR_ADD_OPTS.Y_V,PT_CAMERA_COOR_ADD_OPTS.CLASS FROM PT_CAMERA_COOR,PT_CAMERA_COOR_ADD_OPTS,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID IN ('8') AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
        cursor.execute(string)
        results = cursor.fetchall()
        
        
        x = map(list, list(results))
        x = list(x)
        n=np.asarray(x,dtype=np.float32)
        lm = Lane_Match(dict())
        ground_truth = OrderedDict()
        predicted_path = OrderedDict()
        db.close()
        results_dict_target = dict()
        results_dict_predicted = dict()
        
        real_frame_dict = get_real_frames() 
        prediction_frame_dict = dict()
        
        for t in range(int(n[0,4]),int(n[n.shape[0]-1,4]+1)):
        #for t in range(int(n[0,4]) , int(n[0,4]) + 20):
            #print('original range',int(n[0,4]),int(n[n.shape[0]-1,4]+1))
            w1 = np.where((n[:,4] == t))
            density = len(w1[0])
            agent_list = []
            for objs in n[w1[0],:]:
                agent_list.append([objs[2],objs[3],objs[5],objs[6],objs[7],objs[8],objs[1]])
                #agent_list.append(x,y,theta,x_v,y_v,class,id,cluster_id)
            agent_list = sorted(agent_list, key = order_by)
            
            agent_cluster_map = dict()
            agent_path_map,head_dist = dict(),dict()
            ground_truth[t] = OrderedDict()
            for agents in agent_list:
                agent_id = int(agents[6])
                agent_class = str(agents[5])
                if t == 106:
                    deb = True
                if has_key(ground_truth, agent_id,t):
                    ground_truth[t][agent_id] = ground_truth[t-1][agent_id] + [(agents[0],agents[1])]
                    predicted_path[str(agent_id)+'-'+str(t)] = [(agents[0],agents[1])]
                else:
                    ground_truth[t][agent_id] = [(agents[0],agents[1])]
                    predicted_path[str(agent_id)+'-'+str(t)] = [(agents[0],agents[1])]
                
                if t in results_dict_target.keys():
                    results_dict_target[t][str(agent_id) +'#'+ agent_class] = (agents[0],agents[1])
                else:
                    results_dict_target[t] = dict()
                    results_dict_target[t][str(agent_id) +'#'+ agent_class] = (agents[0],agents[1])
    
                cluster,_,_ = lm.match2([ground_truth[t][agent_id]],str(int(float(agent_class))),True)
                '''if len(cluster) == 0 :
                    cluster,M,dummy = lm.match([ground_truth[t][agent_id]],(agent_id,t),True)
                    cluster = list(cluster.keys())'''
                #cluster_id = list(cluster.keys())[0]
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
                
            
            
            
            prediction_at_current_ts = OrderedDict()
            prediction_frame_dict[t] = dict()
            for agents in agent_list:
                agent_id = int(agents[6])
                prediction_at_current_ts[int(agents[6])] = []
                if t in real_frame_dict.keys():
                    if t in prediction_frame_dict[t].keys():
                        prediction_frame_dict[t][t][agent_id] = dict(real_frame_dict[t][agent_id])
                    else:
                        prediction_frame_dict[t][t] = dict()
                        prediction_frame_dict[t][t][agent_id] = dict(real_frame_dict[t][agent_id])
                        
            
            
            for window in range(90):
                #print('window',window)
                cnt = 0
                skip = False
                for agents in agent_list:
                    density_value = len(agent_list)
                    agent_id = int(agents[6])
                    agent_class = str(agents[5])
                    head = prediction_at_current_ts[agent_id]
                    tail = list(ground_truth[t][int(agents[6])])
                    if prediction_at_current_ts[agent_id] is None:
                        trail = list(ground_truth[t][int(agents[6])])
                    else:
                        trail = list(ground_truth[t][int(agents[6])]) + prediction_at_current_ts[agent_id]
                    
                    if t+window not in prediction_frame_dict[t].keys() or agent_id not in prediction_frame_dict[t][t+window].keys():
                        continue
                    if 'target' not in prediction_frame_dict[t][t+window][agent_id].keys():
                        target_vector = None
                    else:
                        target_vector = prediction_frame_dict[t][t+window][agent_id]['target'] # 2 X 1
                    if target_vector is None or target_vector.size < 2:
                        continue
                    ego_vector = prediction_frame_dict[t][t+window][agent_id]['ego'] # 7 X 1
                    env_vector = prediction_frame_dict[t][t+window][agent_id]['env'] # 1 X 20
                    
                    
                    
                    feed_dict = {data_env_node: env_vector,
                                    data_ego_node: ego_vector,
                                        target_node : target_vector}
                    
                    #print('shapes', env_vector.shape,ego_vector.shape,target_vector.shape)
                    #print('sizes', env_vector.size,ego_vector.size,target_vector.size)
                    output = sess.run([output_vector], feed_dict=feed_dict)
                    
                    x_a,y_a = output[0][0] , output[0][1]
                    
                    ''' uncomment following 2 lines for dynamically changing cluster id with prediction'''
                    cluster,M,dummy = lm.match2([trail],str(int(float(agent_class))),True)
                    '''if len(cluster) == 0 :
                        cluster,M,dummy = lm.match([ground_truth[t][agent_id]],(agent_id,t),True)
                        cluster = list(cluster.keys())'''
                    new_cluster_id = cluster[0]
                    #new_cluster_id = agent_cluster_map[agent_id]
                    
                    new_theta = agents[2]
                    
                    if t == 106:
                        deb = 5
                    disp_x = (ego_vector.astype(float)[2] * (1/fps)) + (0.5 * float(x_a) * (1/fps ** 2))
                    disp_y = (ego_vector.astype(float)[3] * (1/fps)) + (0.5 * float(y_a) * (1/fps ** 2))
                    
                    disp = math.sqrt(disp_x**2 + disp_y**2)
                    
                    
                    '''
                    find next x,y in the direction of agent_path_map[agent_id]
                    '''
                    path = agent_path_map[agent_id]
                    new_points = get_projected_point_2(disp, path , (ego_vector.astype(float)[0] , ego_vector.astype(float)[1]))
                    if new_points == None:
                        #print('no path found at t',t,'w',window,'for agent',agent_id)
                        continue
                        #new_x = (ego_vector.astype(float)[0] + disp * math.cos(new_theta))[0]
                        #new_y = (ego_vector.astype(float)[1] + disp * math.sin(new_theta))[0]
                    else:
                        #print('new points',new_points, len(tail))
                        new_x,new_y = new_points[0],new_points[1]
                    
                    trail.append((new_x,new_y))
                    new_x_v = ego_vector[2] + float(x_a) * (1/fps)
                    new_y_v = ego_vector[3] + float(y_a) * (1/fps)
                    
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
                    
                    
                    #print(new_x  , new_y  ,new_x_v  , new_y_v  , agents[5],  new_cluster_id  , density_value )
                    new_ego_vector = np.reshape(np.asarray([new_x  , new_y  ,new_x_v  , new_y_v  , agents[5],  new_cluster_id  , density_value ]) , newshape = (7,1))
                    if t+window in real_frame_dict.keys():
                        if agent_id in real_frame_dict[t+window].keys():
                            if 'target' in real_frame_dict[t+window][agent_id].keys():
                                new_target_vector = np.copy(real_frame_dict[t+window][agent_id]['target'])
                    else:
                        new_target_vector = None 
                    
                    if t+window+1 in prediction_frame_dict[t].keys():
                        prediction_frame_dict[t][t+window+1][agent_id] = dict()
                        prediction_frame_dict[t][t+window+1][agent_id]['ego'] = np.copy(new_ego_vector)
                        prediction_frame_dict[t][t+window+1][agent_id]['target'] = np.copy(new_target_vector)
                    else:
                        prediction_frame_dict[t][t+window+1] = dict()
                        prediction_frame_dict[t][t+window+1][agent_id] = dict()
                        prediction_frame_dict[t][t+window+1][agent_id]['ego'] = np.copy(new_ego_vector)
                        prediction_frame_dict[t][t+window+1][agent_id]['target'] = np.copy(new_target_vector)
                    
                        
                for agents in agent_list:
                    if t+window+1 not in prediction_frame_dict[t].keys():
                        break
                    agent_id = agents[6]
                    env_batch_slice = np.zeros(shape=(5,6,7), dtype = np.float32)
                    obj_ind = 0
                    for agents_2 in agent_list:
                        env_agent_id = agents_2[6].astype(int)
                        if env_agent_id != agent_id:
                            if env_agent_id not in prediction_frame_dict[t][t+window+1].keys():
                                continue
                            agent_s_cluster = int(prediction_frame_dict[t][t+window+1][env_agent_id]['ego'][5])
                            if agent_s_cluster < 5:
                                env_batch_slice[agent_s_cluster,obj_ind] = np.reshape(np.copy(prediction_frame_dict[t][t+window+1][env_agent_id]['ego']) , newshape = (7) )
                                obj_ind = obj_ind + 1
                    new_env_vector = env_summary.encode_env(env_batch_slice)
                    if t+window+1 not in prediction_frame_dict[t].keys():
                        continue
                    if agent_id not in prediction_frame_dict[t][t+window+1].keys():
                        continue
                    else:
                        prediction_frame_dict[t][t+window+1][agent_id]['env'] = np.copy(new_env_vector)
            
            for agents in agent_list:
                agent_id = int(agents[6])
                predicted_path[str(agent_id)+'-'+str(t)] = predicted_path[str(agent_id)+'-'+str(t)] + \
                                                        prediction_at_current_ts[agent_id] 
                    
            print('-------- : ',t)
        pickle.dump(ground_truth, open( "save_ground_truth_nn_8.p", "wb" ))
        pickle.dump(predicted_path, open( "save_predicted_path_nn_8.p", "wb" ))
        
        print_summary(results_dict_target,results_dict_predicted)
        
        sess.close()
    
            
if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])        
        
