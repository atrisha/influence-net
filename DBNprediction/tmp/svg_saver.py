from matplotlib import pyplot as plt
from matplotlib import animation
import random
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import pickle

fig = plt.figure()

ax = plt.axes(xlim=(-9, 9), ylim=(-5, 50))
color_map = {'o1': 'k-', 'o2':'b-' , 'o3' : 'g-'}
N = 4

gt = pickle.load( open( "../runs/NN/save_ground_truth_nn_10.p", "rb" ) )
pp = pickle.load( open( "../runs/NN/save_predicted_path_nn_10.p", "rb" ) )

min_t = min([int(i) for i in gt.keys()])
w = .1
fps = 30
l = 31
jump_ind = 3
lines = []
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=127, vmax=144)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)    
c_m = {'o1':0,'o2':1,'o3':2}

for k,v in gt.items():
    plt.xlim(-10,10)
    plt.ylim(-5,55)
    for objs,l in v.items():
        current_loc_x = l[-1][0]
        current_loc_y = l[-1][1]
        plt.plot([current_loc_x],current_loc_y,'ro')
        plt.xlim(-10,10)
        plt.ylim(-5,55)
        pp_key = str(objs)+'-'+str(k)
        fp_x,fp_y = [],[]
        for i,f_lc in enumerate(pp[pp_key]):
            if i<30 :
                fp_x.append(f_lc[0])
                fp_y.append(f_lc[1])
            else:
                break
        plt.plot(fp_x,fp_y,'b-')
    plt.savefig("./movie/test"+str(k)+".svg")
    print(k)
    plt.clf()
