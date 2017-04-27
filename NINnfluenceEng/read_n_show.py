'''
Created on Feb 23, 2017

@author: atri
'''


import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

nn_v_10 = pickle.load( open( "save_result_summary_vehicles_nn.p", "rb" ) )
dbn_v_10 = pickle.load( open( "save_result_summary_vehicles_dbn_10.p", "rb" ) )

nn_p_8 = pickle.load( open( "save_result_summary_pedestrians_nn_8.p", "rb" ) )
nn_v_8 = pickle.load( open( "save_result_summary_vehicles_nn_8.p", "rb" ) )

dbn_p_8 = pickle.load( open( "save_result_summary_pedestrians_dbn_8.p", "rb" ) )
dbn_v_8 = pickle.load( open( "save_result_summary_vehicles_dbn_8.p", "rb" ) )


for k,v in nn_v_8.items():
    if k in nn_v_10.keys():
        nn_v_8[k] = v+nn_v_10[k]
        
for k,v in dbn_v_8.items():
    if k in dbn_v_10.keys():
        dbn_v_8[k] = v+dbn_v_10[k]        

v_nn_x,p_nn_x,v_dbn_x,p_dbn_x=[],[],[],[]
v_nn_y,p_nn_y,v_dbn_y,p_dbn_y,v_nn_sd=[],[],[],[],[]
data_x = {'ped_dbn' : [] , 'ped_nn' : [] , 'veh_dbn' : [] , 'veh_nn' : [] }
data_y = {'ped_dbn' : [] , 'ped_nn' : [] , 'veh_dbn' : [] , 'veh_nn' : [] }
data_sd = {'ped_dbn' : [] , 'ped_nn' : [] , 'veh_dbn' : [] , 'veh_nn' : [] }

for k,v in dbn_p_8.items():
    data_x['ped_dbn'] = data_x['ped_dbn'] + [float(k)*0.03]
    #x1.append(float(k)*0.03)
    data_y['ped_dbn'] = data_y['ped_dbn'] + [np.mean(v)]
    #y1.append(np.mean(v))
    data_sd['ped_dbn'] = data_sd['ped_dbn'] + [np.std(v)]

for k,v in nn_p_8.items():
    data_x['ped_nn'] = data_x['ped_nn'] + [float(k)*0.03]
    #x1.append(float(k)*0.03)
    data_y['ped_nn'] = data_y['ped_nn'] + [np.mean(v)]
    #y1.append(np.mean(v))
    data_sd['ped_nn'] = data_sd['ped_nn'] + [np.std(v)]
    
for k,v in dbn_v_8.items():
    data_x['veh_dbn'] = data_x['veh_dbn'] + [float(k)*0.03]
    #x1.append(float(k)*0.03)
    data_y['veh_dbn'] = data_y['veh_dbn'] + [np.mean(v)]
    #y1.append(np.mean(v))
    data_sd['veh_dbn'] = data_sd['veh_dbn'] + [np.std(v)]

for k,v in nn_v_8.items():
    data_x['veh_nn'] = data_x['veh_nn'] + [float(k)*0.03]
    #x1.append(float(k)*0.03)
    data_y['veh_nn'] = data_y['veh_nn'] + [np.mean(v)]
    #y1.append(np.mean(v))
    data_sd['veh_nn'] = data_sd['veh_nn'] + [np.std(v)]

'''for k,v in nn_p_8.items():
    x.append(float(k)*0.03)
    y.append(np.mean(v))
    sd.append(np.std(v))
    
for k,v in nn_v_8.items():
    xv.append(float(k)*0.03)
    yv.append(np.mean(v))
    sd.append(np.std(v))    '''

gain = [((e1-e2)/e1) * 100 for e2,e1 in zip(data_y['ped_nn'],data_y['ped_dbn'])]
print('mean percentage improvement of influence-net over dbn for pedestrians:' , np.mean(gain))
gain = [((e1-e2)/e1) * 100 for e2,e1 in zip(data_y['veh_nn'],data_y['veh_dbn'])]
print('mean percentage improvement of influence-net over dbn for vehicles:' , np.mean(gain))

f, axarr = plt.subplots(2, 2)    
axarr[0, 0].errorbar(data_x['ped_nn'], data_y['ped_nn'], data_sd['ped_nn'], linestyle='None', marker='.', ecolor='0.75',color='k')
axarr[0, 0].set_title('Pedestrians (Influence Network)')
axarr[0, 0].set_ylim(-2,8)
axarr[0, 0].set_xlabel('prediction window (secs.)')
axarr[0, 0].set_ylabel('rms error and standard deviation (meters)')

axarr[0, 1].errorbar(data_x['ped_dbn'], data_y['ped_dbn'], data_sd['ped_dbn'], linestyle='None', marker='.', ecolor='0.75',color='k')
axarr[0, 1].set_title('Pedestrians (Dynamic Bayesian Network)')
axarr[0, 1].set_ylim(-2,8)
axarr[0, 1].set_xlabel('prediction window (secs.)')
axarr[0, 1].set_ylabel('rms error and standard deviation (meters)')

axarr[1, 0].errorbar(data_x['veh_nn'], data_y['veh_nn'], data_sd['veh_nn'], linestyle='None', marker='.', ecolor='0.75',color='k')
axarr[1, 0].set_title('Vehicles (Influence Network)')
axarr[1, 0].set_ylim(-2,8)
axarr[1, 0].set_xlabel('prediction window (secs.)')
axarr[1, 0].set_ylabel('rms error and standard deviation (meters)')

axarr[1, 1].errorbar(data_x['veh_dbn'], data_y['veh_dbn'], data_sd['veh_dbn'], linestyle='None', marker='.', ecolor='0.75',color='k')
axarr[1, 1].set_title('Vehicles (Dynamic Bayesian Network)')
axarr[1, 1].set_ylim(-2,8)
axarr[1, 1].set_xlabel('prediction window (secs.)')
axarr[1, 1].set_ylabel('rms error and standard deviation (meters)')


plt.show()