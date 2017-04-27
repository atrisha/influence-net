'''
Created on Dec 2, 2016

@author: atri
'''

from trajectory_cluster import *
import numpy as np
from particle_sample import *
from sklearn.ensemble import RandomForestRegressor
import math
import numpy.ma as ma
import sys
import pickle
np.set_printoptions(threshold=np.nan)
def order_by(p):
    return p[0]
    

def flatten_sample(elem):
    elem = sorted(elem, key = order_by)
    r = []
    a = []
    for i in elem:
        t = list(i)
        a_t = [t[6],t[7],t[8]]
        del t[6]
        del t[6]
        del t[6]
        a = a + a_t
        r = r + t
    return [r,a]       
     
def has_converged(delta):
    for i in range(delta.shape[1]):
        delta_slice_mean = delta[:,i,0]
        

def sample(sampling_tuple, merged_data_trajectory, merged_particle_ids, C):
    particle_list = particle_sample(sampling_tuple, merged_data_trajectory, merged_particle_ids, C)
    len_dict = dict()
    for i in particle_list:
        l = len(i)
        if l in len_dict.keys():
            len_dict[l].append(i)
        else :
            len_dict[l] = [i]
    
    for k,v in len_dict.items():
        len_dict[k] = [flatten_sample(elem) for elem in v]
    
    return len_dict

def append_samples(a,b):
    for k,v in a.items():
        if k in b.keys():
            a[k] = a[k] + b[k]
    for k,v in b.items():
        if k not in a.keys():
            a[k] = b[k]        
    return a


def create_model():
    merged_data_trajectory,sampling_tuple,M,C,merged_particle_ids = cluster()
    last_model_params = np.empty([10,30,2])
    last_model_params.fill(sys.maxsize)
    
    all_samples = dict()
    ''' first guess of the model'''
    len_dict = sample(sampling_tuple, merged_data_trajectory, merged_particle_ids, C)
    all_samples = append_samples(all_samples, len_dict)
    model_params = np.empty([10,30,2])
    model_params.fill(sys.maxsize)
    
    mean_functions = np.empty(shape=(10,30,1),dtype=object)
    
    lm1 = []
    for md in merged_data_trajectory:
        lm1 = lm1 + [md[9]]
    min_lm = min(lm1)
    max_lm = max(lm1)
    
    
    for density in range(model_params.shape[0]):
        if density in len_dict.keys() and density > 0:
            x_data,y_data = [],[]
            for i in len_dict[density]:
                x_data.append(i[0])
            x_data_array = np.array(x_data).astype("f4")
            #print(np.any(np.isnan(x_data_array)))
            #print(np.all(np.isfinite(x_data_array)))
            y_data = []
            for i in len_dict[density]:
                y_data.append(i[1])
            y_data_array = np.array(y_data).astype("f4")  
            #print(np.any(np.isnan(y_data_array)))
            #print(np.all(np.isfinite(y_data_array)))
            for target_index in range(3*density):            
                target = np.asarray(y_data_array[:,target_index], dtype="f4")
                target = target.astype("f4")
                train = x_data_array
                rf = RandomForestRegressor(n_estimators=100, min_samples_split=10)
                rf.fit(train, target)
                mean_functions[density,target_index,0] = pickle.dumps(rf)
                res = rf.predict(train).astype("f4")
                mean = np.mean(res)
                variance = np.var(res)
                model_params[density,target_index,0] = mean
                model_params[density,target_index,1] = variance
                
    for loop in range(100):
        len_dict = sample(sampling_tuple, merged_data_trajectory, merged_particle_ids, C)
        for density in range(model_params.shape[0]):
            if density in len_dict.keys() and density > 0:
                x_data,y_data = [],[]
                for i in len_dict[density]:
                    x_data.append(i[0])
                x_data_array = np.array(x_data).astype("f4")
                #print(np.any(np.isnan(x_data_array)))
                #print(np.all(np.isfinite(x_data_array)))
                y_data = []
                for i in len_dict[density]:
                    y_data.append(i[1])
                y_data_array = np.array(y_data).astype("f4")
                #print(np.any(np.isnan(y_data_array)))
                #print(np.all(np.isfinite(y_data_array)))
                for target_index in range(3*density):
                    
                    
                    ''' Expectation '''
                    s2 = mean_functions[density,target_index,0]
                    if s2 is None:
                        print('New created in iteration')
                        curr_rf_model = RandomForestRegressor(n_estimators=100, min_samples_split=10)
                        target = np.asarray(y_data_array[:,target_index], dtype="f4")
                        train = x_data_array
                        curr_rf_model.fit(train, target)
                    else:
                        curr_rf_model = pickle.loads(s2)
                    estimate_means = curr_rf_model.predict(x_data_array).astype("f4")   
                    estimate_variance = np.var(estimate_means)
                    sd = abs(math.sqrt(estimate_variance))
                    actual_estimates = [np.random.normal(m, sd) for m in estimate_means] if sd > 0 else estimate_means
                    
                    ''' Maximization '''
                    
                    cumul_x_data,cumul_y_data = [],[]
                    if density in all_samples.keys():
                        for i in all_samples[density]:
                            cumul_x_data.append(i[0])
                        cumul_x_data_array = np.array(cumul_x_data).astype("f4")
                        cumul_x = np.concatenate((cumul_x_data_array, x_data_array))
                        
                        for i in all_samples[density]:
                            cumul_y_data.append(i[1][target_index])
                        cumul_y_data_array = np.array(cumul_y_data).astype("f4")
                        cumul_y = np.concatenate((cumul_y_data_array, np.asarray(actual_estimates, dtype="f4")))
                    else:
                        cumul_x = x_data_array
                        cumul_y = np.asarray(actual_estimates, dtype="f4")
                            
                    target = cumul_y
                    train = cumul_x        
                    
                    
                    rf = RandomForestRegressor(n_estimators=100, min_samples_split=10)
                    rf.fit(train, target)
                    res = rf.predict(train)
                    mean = np.mean(res)
                    variance = np.var(res)
                    
                    model_params[density,target_index,0] = mean
                    model_params[density,target_index,1] = variance
                    
                    mean_functions[density,target_index,0] = pickle.dumps(rf)
        
        all_samples = append_samples(all_samples, len_dict)
        
        if loop > 1:
            delta_model_params = abs(ma.masked_values(model_params, sys.maxsize) - ma.masked_values(last_model_params, sys.maxsize))
            delta_x_a_mean = delta_model_params[:,0:27:3,0].max()
            delta_y_a_mean = delta_model_params[:,1:28:3,0].max()
            #print(delta_x_a_mean,delta_y_a_mean)
        last_model_params = np.copy(model_params)        
        
        print("-loop",loop)
        print()
        print("----------------------")
    
    func_2_dump = np.empty(shape=(10,30,1),dtype=object)
    for i in range(10):
        for j in range(3*10):
            s2 = mean_functions[i,j,0]
            if s2 is not None:
                func_2_dump[i,j,0] = pickle.loads(s2)
            else:
                func_2_dump[i,j,0] = None
    pickle.dump(func_2_dump, open( "save_mean_functions.p", "wb" ))
    pickle.dump(model_params, open( "save_model_params.p", "wb" ))
    return mean_functions,model_params


if __name__ == '__main__':
    create_model()   
     