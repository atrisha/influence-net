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

env_data_batch, ego_data_batch,target_data_batch,(max_density_index,density) = nn_engine_input.get_batch()
print(max_density_index,density)
print(env_data_batch.shape)
x_max,x_min,y_max,y_min = 20,-20,55,-5
env_slice = env_data_batch[max_density_index]
cnt = 0
env_dict = {0:[],1:[],2:[],3:[],4:[]}

for clusters in range(env_slice.shape[0]):
    data_x_v = np.zeros(shape=(40, 60))
    x_l,y_l,x_v_l,y_v_l = [],[],[],[]
    found = False
    for agents in range(env_slice.shape[1]):
        x,y,x_v,y_v = env_slice[clusters,agents,0],env_slice[clusters,agents,1],env_slice[clusters,agents,2],env_slice[clusters,agents,3]
        if x != 0 or y !=0 or x_v !=0 or y_v!=0 :
            found = True
            x_l.append(x)
            y_l.append(y)
            x_v_l.append(x_v)
            y_v_l.append(y_v)
            
            data_x_v[math.floor(x),math.floor(y)] = math.sqrt(x_v**2 + y_v**2) 
    if not found :
        print('nothing for cluster ',clusters)
        continue
    else:
        print('showing cluster ',clusters, 'with density', len(x_l))
        #f, axarr = plt.subplots(4)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_axis = np.asarray(list(range(x_min,x_max)))
        y_axis = np.asarray(list(range(y_min,y_max)))
        
        
        
        spl = sp.RectBivariateSpline(x=x_axis, y=y_axis ,z = data_x_v)
        x_data,y_data,z_data = [],[],[]
        for i in x_axis:
            for j in y_axis:
                x_data.append(i)
                y_data.append(j)
                z_data.append(spl.ev(xi = i, yi = j))
        ax.plot_wireframe(x_data,y_data,z_data, rstride=1, cstride=1)
        
        print(spl.get_coeffs().shape)
        plt.show()        
                
        '''x_l_extn = [[],[],[]]
        for i in range(x_min,x_max):
            x_l_extn[0].append(i)
            x_l_extn[1].append(0)
            x_l_extn[2].append(0)
            for ind,elem in enumerate(x_l):
                if i < elem <= i+1 :
                    x_l_extn[0].append(elem)
                    x_l_extn[1].append(x_v_l[ind])
                    x_l_extn[2].append(y_v_l[ind])
        y_l_extn = [[],[],[]]
        for i in range(y_min,y_max):
            y_l_extn[0].append(i)
            y_l_extn[1].append(0)
            y_l_extn[2].append(0)
            for ind,elem in enumerate(y_l):
                if i < elem <= i+1 :
                    y_l_extn[0].append(elem)
                    y_l_extn[1].append(x_v_l[ind])
                    y_l_extn[2].append(y_v_l[ind])
        x_axis = np.asarray(x_l_extn[0])
        y_axis = np.asarray(y_l_extn[0])
        print(x_axis.shape,y_axis.shape)
        data_x_v = np.zeros(shape=(x_axis.shape[0], y_axis.shape[0]))
        for i in range(x_axis.shape[0]):
            for j in range(y_axis.shape[0]):
                data_x_v[i,j] = x_l_extn[1][i],y_l_extn[1][j]
        print(data_x_v.shape)'''
         
    '''                
    if len(x_l_extn[0]) is not None:
        z1 = np.poly1d(np.polyfit(x_l_extn[0], x_l_extn[1], 3))
        z2 = np.poly1d(np.polyfit(x_l_extn[0], x_l_extn[2], 3))
    if len(y_l_extn[0]) is not None:
        z3 = np.poly1d(np.polyfit(y_l_extn[0], y_l_extn[1], 3))
        z4 = np.poly1d(np.polyfit(y_l_extn[0], y_l_extn[2], 3))
        
    print('x - x_v (co-effs)',z1)
    print('values',y_l_extn[0],  z1(y_l_extn[1]))
    pl_x,pl_y = np.arange(x_min,x_max) , np.poly1d(z1)
    axarr[0].plot(y_l_extn[0],  z1(y_l_extn[1]) )
    axarr[0].set_title('x with x_v')
    
    print('x - y_v (co-effs)',z2)
    pl_x,pl_y = np.arange(x_min,x_max) , np.poly1d(z2)
    axarr[1].plot(y_l_extn[0],  z2(y_l_extn[2]) )
    axarr[1].set_title('x with y_v')
    
    print('y - x_v (co-effs)',z3)
    pl_x,pl_y = np.arange(y_min,y_max) , np.poly1d(z3)
    axarr[2].plot(y_l_extn[0],  z3(y_l_extn[1]) )
    axarr[2].set_title('y with x_v')
    
    print('y - y_v (co-effs)',z4)
    pl_x,pl_y = np.arange(y_min,y_max) , np.poly1d(z4)
    axarr[3].plot(y_l_extn[0],  z4(y_l_extn[2]) )
    axarr[3].set_title('y with y_v') 
    print('-----')
    plt.show()
    '''
        
