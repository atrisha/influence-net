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

gt = pickle.load( open( "../runs/with-path/save_ground_truth.p", "rb" ) )
pp = pickle.load( open( "../runs/with-path/save_predicted_path.p", "rb" ) )

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
def init():
    line_new = plt.plot([], [], color='k', linestyle='-')[0]
    line_new.set_data([-6,-14], [19,50])
    line_new = plt.plot([], [], color='k', linestyle='-')[0]
    line_new.set_data([-6,-8], [19,15])
    line_new = plt.plot([], [], color='gray', linestyle='-')[0]
    line_new.set_data([-7.3,1.6], [49,18])
    line_new = plt.plot([], [], color='lightgray', linestyle='--')[0]
    line_new.set_data([-11,-4], [50,23])
    return lines

def animate(i):
    
    grnd_ind = math.floor(((i+1) * fps*w) /  l)*jump_ind + min_t
    pred_ind = int((i % max( math.floor(l / (fps * w)) , 1 ) ) * (fps * w))
    pred_truth_ind = grnd_ind + pred_ind
    print('indexes',i, grnd_ind, pred_ind,math.floor(((i+1) * fps*w) /  l))
    if i == 3310:
        deb = True
    lines = []
    lines_p = []
    for k in gt[grnd_ind].keys():
        line_new = plt.plot([], [], color=scalarMap.to_rgba(k), marker = 'o', linestyle='-')[0]
        o_id = k
        key = str(o_id) + '-' + str(grnd_ind)
        '''print(key)'''
        pts_list_x_p = [k1[0] for k1 in pp[key][:pred_ind+1]]
        pts_list_y_p = [k1[1] for k1 in pp[key][:pred_ind+1]] 
        if k in gt[pred_truth_ind].keys():
            pts_list_x_p_t = [k1[0] for k1 in gt[pred_truth_ind][k]]
            pts_list_y_p_t = [k1[1] for k1 in gt[pred_truth_ind][k]]
        
        pts_list_x = [k1[0] for k1 in gt[grnd_ind][k]]
        pts_list_y = [k1[1] for k1 in gt[grnd_ind][k]]
        line_new.set_data(pts_list_x,pts_list_y)
        line_p_new = plt.plot([], [], color=scalarMap.to_rgba(k), linestyle='-')[0]
        line_p_new.set_data(pts_list_x_p,pts_list_y_p)
        if k in gt[pred_truth_ind].keys():
            line_p_t_new = plt.plot([], [], color='r', linestyle='-')[0]
            line_p_t_new.set_data(pts_list_x_p_t,pts_list_y_p_t)
            lines.append(line_p_t_new)
        lines.append(line_p_new)
        lines.append(line_new)
        
        '''print('lines_g',pts_list_x,pts_list_y)
        print('lines_p',pts_list_x_p,pts_list_y_p)'''
    plt.savefig("test"+str(i)+".svg")
    return lines

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=25000, interval=100, blit=True)

plt.show()