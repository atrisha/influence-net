'''
Created on Dec 19, 2016

@author: atri
'''

import numpy as np
import pickle
from collections import OrderedDict
from scipy.spatial import distance
import db_helper as helper

def order_by(p):
    return p[0]

mean_functions = pickle.load( open( "../DBNprediction/save_mean_functions.p", "rb" ) )
model_params = pickle.load( open( "../DBNprediction/save_model_params.p", "rb" ) )
    
    

def predict():
    fps = 30
    db = helper.connect()
    cursor = db.cursor()
    string = """SELECT PT_CAMERA_COOR.ID,PT_CAMERA_COOR.PID,PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR.T,PT_CAMERA_COOR_ADD_OPTS.THETA,
                    PT_CAMERA_COOR_ADD_OPTS.X_V,PT_CAMERA_COOR_ADD_OPTS.Y_V,PT_CAMERA_COOR_ADD_OPTS.CLASS,PT_CAMERA_COOR_ADD_OPTS.CLUSTER,OBJECT.ID FROM PT_CAMERA_COOR,
                        PT_CAMERA_COOR_ADD_OPTS,OBJECT,ANNOTATION WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.ID IN ('10') 
                            AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"""
    cursor.execute(string)
    results = cursor.fetchall()
    

    
    print(mean_functions.shape,model_params.shape)
    
    
        
    
    x = map(list, list(results))
    x = list(x)
    n=np.asarray(x,dtype=np.float32)
    db.close()
    sum_loss = []
    t_list = []
    for t in range(int(n[0,4]),int(n[n.shape[0]-1,4]+1)):
    #for t in range(1135 , 1145):
        c2 = n[:,4]==t
        c3 = n[:,4]==t+1
        r1 = n[c2]
        r2 = n[c3]
        object_ids_r1 = r1[:,10]
        object_ids_r2 = r2[:,10]
    
        if np.array_equal(object_ids_r1, object_ids_r2):
            t_list.append((t,t+1))
    for ind,t in enumerate(t_list):        
        print('t - ',t)
        w1 = np.where((n[:,4] == t[0]))
        w2 = np.where((n[:,4] == t[1]))
        density = len(w1[0])
        prediction_input = []
        agent_list,agent_list_2 = [],[]
        for objs in n[w1[0],:]:
            agent_list.append([objs[2],objs[3],objs[5],objs[6],objs[7],objs[8],objs[1],objs[9]])
        
        for objs in n[w2[0],:]:
            agent_list_2.append([objs[2],objs[3],objs[5],objs[6],objs[7],objs[8],objs[1],objs[9]])
            #agent_list.append(x,y,theta,x_v,y_v,class,id,cluster_id)
        agent_list = sorted(agent_list, key = order_by)
        agent_list_2 = sorted(agent_list_2, key = order_by)
        
        for agents in agent_list:
            if t == 106:
                deb = True
            prediction_input = prediction_input + [agents[0],agents[1],agents[2],agents[3],agents[4],agents[5],agents[7]]
            
        prediction_input = np.asarray(prediction_input).reshape(1,-1)
        
        for cnt,agents in enumerate(agent_list):    
            predictor_x_a = mean_functions[density,cnt*3,0]
            predictor_y_a = mean_functions[density,cnt*3 + 1,0]
            predictor_omega = mean_functions[density,cnt*3 + 2,0]
            prediction_input = prediction_input.astype(np.float32)
            x_a = np.random.normal(loc = predictor_x_a.predict(prediction_input) , scale = np.sqrt(model_params[density,cnt*3,1]))
            y_a = np.random.normal(loc = predictor_y_a.predict(prediction_input) , scale = np.sqrt(model_params[density,cnt*3 + 1,1]))
            target_x_a = float(agent_list_2[cnt][3]) - float(agents[3])
            target_y_a = float(agent_list_2[cnt][4]) - float(agents[4])
            
            loss = np.asarray([(target_x_a - x_a) ** 2 , (target_y_a - y_a) ** 2])
            sum_loss.append(loss) 
            cnt = cnt + 1
    print('loss statistics after dbn run :', 'mean=',np.mean(sum_loss),' median=',np.median(sum_loss),' max=',np.max(sum_loss),' min=',np.min(sum_loss))        
if __name__ == '__main__':
    predict()    
    
