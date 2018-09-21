'''
Created on Dec 20, 2016

@author: atrisha
'''

import pickle
import math
import numpy as np

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
gt = pickle.load( open( "./runs/without-path/save_ground_truth.p", "rb" ) )
pp = pickle.load( open( "./runs/without-path/save_predicted_path.p", "rb" ) )

mean_functions = pickle.load( open( "save_mean_functions.p", "rb" ) )
model_params = pickle.load( open( "save_model_params.p", "rb" ) )

prediction_input = [3.54665294008 , 10.6424537368 , 2.82106433902, -2.73, 0.0, 1.0, 1]
prediction_input = np.asarray(prediction_input).reshape(1,-1)

predictor_x_a = mean_functions[1,0*3,0]
predictor_y_a = mean_functions[1,0*3 + 1,0]
predictor_omega = mean_functions[1,0*3 + 2,0]
prediction_input = prediction_input.astype(np.float32)
x_a = np.random.normal(loc = predictor_x_a.predict(prediction_input) , scale = np.sqrt(model_params[1,0*3,1]))
y_a = np.random.normal(loc = predictor_y_a.predict(prediction_input) , scale = np.sqrt(model_params[1,0*3 + 1,1]))

disp_x = (prediction_input[0,0*7 + 3] * (1/30)) + (0.5 * float(x_a) * math.pow((1/30), 2))
disp_y = (prediction_input[0,0*7 + 4] * (1/30)) + (0.5 * float(y_a) * math.pow((1/30), 2))

disp = math.pow((math.pow(disp_x,2) + math.pow(disp_y, 2)),0.5)

''' window , agent_id , t '''
#got_theta = get_projected_points(cluster_details,new_cluster_id,disp,trail,t,agent_id,window,head,tail)
got_theta = None
if got_theta is not None:
    new_theta = got_theta
else:
    omega = np.random.normal(loc = predictor_omega.predict(prediction_input) , scale = np.sqrt(model_params[1,0*3 + 2,1]))
    # Draw from distribution
    sin_new_theta = modulo_add(omega , np.sin(prediction_input[0,0*7 + 2]))
    new_theta = get_theta_from_sine(disp_x,disp_y,sin_new_theta)
    #print(omega + np.sin(prediction_input[0,cnt*7 + 2]), new_theta)

new_theta_deg = np.rad2deg(new_theta)
new_x = prediction_input[0,0*7] + disp * math.cos(new_theta)
new_y = prediction_input[0,0*7 + 1] + disp * math.sin(new_theta)
                

print(new_x,',',new_y,',',new_theta)
print()