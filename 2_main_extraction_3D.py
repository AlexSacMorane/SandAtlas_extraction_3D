#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

from libtiff import TIFF
from pathlib import Path
import shutil, os, pickle, skfmm
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

def create_folder(name):
    '''
    Create a new folder. If it already exists, it is erased.
    '''
    if Path(name).exists():
        shutil.rmtree(name)
    os.mkdir(name)

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

# file to read
namefile = 'SandAtlas_Hamburg_labelled.tif'

# extraction zone
i_x_min = 400
i_x_max = 460
i_y_min = 400
i_y_max = 460
i_z_min = 400
i_z_max = 460

# initialization
map_data = np.array(np.zeros((i_x_max-i_x_min+1,
                              i_y_max-i_y_min+1,
                              i_z_max-i_z_min+1)))

# margins 
margins = 5

# mesh size
m_size = 11 # µm

# number of grains by data set
n_grain_data = 20
i_data = 1 

# volume minimum
min_volume = 20

# reduce sdf 
reduce_sdf = False

# create folder
create_folder('data')

#-------------------------------------------------------------------------------
# Read data
#-------------------------------------------------------------------------------

# read file
file_tiff = TIFF.open(namefile)
L_image = file_tiff.read_image()

# iterate on x
i = 0
for image in file_tiff.iter_images():
    i = i + 1
    if i_x_min <= i and i <= i_x_max:
        map_data[i-i_x_min, :, :] = image[i_y_min: i_y_max+1, i_z_min: i_z_max+1]

#-------------------------------------------------------------------------------
# Extract labels
#-------------------------------------------------------------------------------

L_label = []
# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on y
    for i_y in range(map_data.shape[1]):
        # iterate on z
        for i_z in range(map_data.shape[2]):
            if map_data[i_x, i_y, i_z] not in L_label and not map_data[i_x, i_y, i_z] == 0:
                L_label.append(map_data[i_x, i_y, i_z])

print(len(L_label), 'total grains')

#-------------------------------------------------------------------------------
# Remove grains on the sides
#-------------------------------------------------------------------------------

L_label_to_remove = []

# -x side
# iterate on y
for i_y in range(map_data.shape[1]):
    # iterate on z
    for i_z in range(map_data.shape[2]):
        if map_data[0, i_y, i_z] not in L_label_to_remove and not map_data[0, i_y, i_z] == 0:
            L_label_to_remove.append(map_data[0, i_y, i_z])

# x side
# iterate on y
for i_y in range(map_data.shape[1]):
    # iterate on z
    for i_z in range(map_data.shape[2]):
        if map_data[-1, i_y, i_z] not in L_label_to_remove and not map_data[-1, i_y, i_z] == 0:
            L_label_to_remove.append(map_data[-1, i_y, i_z])

# -y side
# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on z
    for i_z in range(map_data.shape[2]):
        if map_data[i_x, 0, i_z] not in L_label_to_remove and not map_data[i_x, 0, i_z] == 0:
            L_label_to_remove.append(map_data[i_x, 0, i_z])

# y side
# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on z
    for i_z in range(map_data.shape[2]):
        if map_data[i_x, -1, i_z] not in L_label_to_remove and not map_data[i_x, -1, i_z] == 0:
            L_label_to_remove.append(map_data[i_x, -1, i_z])

# -z side
# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on y
    for i_y in range(map_data.shape[1]):
        if map_data[i_x, i_y, 0] not in L_label_to_remove and not map_data[i_x, i_y, 0] == 0:
            L_label_to_remove.append(map_data[i_x, i_y, 0])

# z side
# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on y
    for i_y in range(map_data.shape[1]):
        if map_data[i_x, i_y, -1] not in L_label_to_remove and not map_data[i_x, i_y, -1] == 0:
            L_label_to_remove.append(map_data[i_x, i_y, -1])

# HARD CODING
L_label_to_remove = []

# iterate on label to remove
for lr in L_label_to_remove:
    L_label.remove(lr)

# iterate on x
for i_x in range(map_data.shape[0]):
    # iterate on y
    for i_y in range(map_data.shape[1]):
        # iterate on z
        for i_z in range(map_data.shape[2]):
            if map_data[i_x, i_y, i_z] in L_label_to_remove:
                map_data[i_x, i_y, i_z] = 0
                
# print
print(len(L_label), 'available grains')

#-------------------------------------------------------------------------------
# Find limits
#-------------------------------------------------------------------------------

# -x limit
i_x = 0
found = False
while (not found) and (i_x < map_data.shape[0]):
    if np.sum(map_data[i_x, :, :]) == 0:
        i_x_min_lim = i_x
    else :
        found = True
    i_x = i_x + 1
i_x_min_lim = 0

# +x limit
i_x = map_data.shape[0]-1
found = False
while not found and 0 <= i_x:
    if np.sum(map_data[i_x, :, :]) == 0:
        i_x_max_lim = i_x
    else :
        found = True
    i_x = i_x - 1
i_x_max_lim = map_data.shape[0]-1

print('x limits:', i_x_min_lim, '-', i_x_max_lim, ' /', map_data.shape[0])

# -y limit
i_y = 0
found = False
while (not found) and (i_y < map_data.shape[1]):
    if np.sum(map_data[:, i_y, :]) == 0:
        i_y_min_lim = i_y
    else :
        found = True
    i_y = i_y + 1
i_y_min_lim = 0

# +y limit
i_y = map_data.shape[1]-1
found = False
while not found and 0 <= i_y:
    if np.sum(map_data[:, i_y, :]) == 0:
        i_y_max_lim = i_y
    else :
        found = True
    i_y = i_y - 1
i_y_max_lim = map_data.shape[1]-1

print('y limits:', i_y_min_lim, '-', i_y_max_lim, ' /', map_data.shape[1])

# -z limit
i_z = 0
found = False
while (not found) and (i_z < map_data.shape[2]):
    if np.sum(map_data[:, :, i_z]) == 0:
        i_z_min_lim = i_z
    else :
        found = True
    i_z = i_z + 1
i_z_min_lim = 0

# +z limit
i_z = map_data.shape[2]-1
found = False
while not found and 0 <= i_z:
    if np.sum(map_data[:, :, i_z]) == 0:
        i_z_max_lim = i_z
    else :
        found = True
    i_z = i_z - 1
i_z_max_lim = map_data.shape[2]-1

print('z limits:', i_z_min_lim, '-', i_z_max_lim, ' /', map_data.shape[2])

#-------------------------------------------------------------------------------
# Adapt data
#-------------------------------------------------------------------------------

map_data = map_data[i_x_min_lim+1:i_x_max_lim, i_y_min_lim+1:i_y_max_lim, i_z_min_lim+1:i_z_max_lim]
print('useful shape:', map_data.shape)

# margin -x
map_margin = np.zeros((margins, map_data.shape[1], map_data.shape[2]))
map_data = np.concatenate((map_margin, map_data), axis=0)
# margin +x
map_data = np.concatenate((map_data, map_margin), axis=0)

# margin -y
map_margin = np.zeros((map_data.shape[0], margins, map_data.shape[2]))
map_data = np.concatenate((map_margin, map_data), axis=1)
# margin +y
map_data = np.concatenate((map_data, map_margin), axis=1)

# margin -z
map_margin = np.zeros((map_data.shape[0], map_data.shape[1], margins))
map_data = np.concatenate((map_margin, map_data), axis=2)
# margin +z
map_data = np.concatenate((map_data, map_margin), axis=2)

print('final shape:', map_data.shape)

#-------------------------------------------------------------------------------
# Compute mesh
#-------------------------------------------------------------------------------

x_L = np.arange(-m_size*(map_data.shape[0]-1)/2, m_size*(map_data.shape[0]-1)/2+0.1*m_size, m_size)
y_L = np.arange(-m_size*(map_data.shape[1]-1)/2, m_size*(map_data.shape[1]-1)/2+0.1*m_size, m_size)
z_L = np.arange(-m_size*(map_data.shape[2]-1)/2, m_size*(map_data.shape[2]-1)/2+0.1*m_size, m_size)

#-------------------------------------------------------------------------------
# Compute wall
#-------------------------------------------------------------------------------

L_pos_w = [[x_L[margins], 0, 0, 0],
           [x_L[-1-margins], 0, 0, 0],
           [0, y_L[margins], 0, 1],
           [0, y_L[-1-margins], 0, 1],
           [0, 0, z_L[margins], 2],
           [0, 0, z_L[-1-margins], 2]]

#-------------------------------------------------------------------------------
# Compute sdfs
#-------------------------------------------------------------------------------

# init
L_sdf = []
L_x_L = []
L_y_L = []
L_z_L = []
L_rbm = []
L_counter = []

# iterate on labels
for i_sdf in range(len(L_label)):
    print(i_sdf+1,'/', len(L_label))

    bin_map = -np.ones(map_data.shape)
    counter = 0
    # iterate on x
    for i_x in range(map_data.shape[0]):
        # iterate on y
        for i_y in range(map_data.shape[1]):
            # iterate on z  
            for i_z in range(map_data.shape[2]):
                # select label
                if map_data[i_x, i_y, i_z] == L_label[i_sdf]:
                    bin_map[i_x, i_y, i_z] = 1
                    counter = counter + 1

    # threshold value on the volume  
    if counter < min_volume :
        print('grain too small')   
        # save data
        if i_sdf==len(L_label)-1:
            # prepare dem_base.py
            dict_save = {
                'L_sdf_i_map': L_sdf,
                'L_x_L': L_x_L,
                'L_y_L': L_y_L,
                'L_z_L': L_z_L,
                'L_rbm': L_rbm
            }
            with open('data/level_set_part'+str(i_data)+'.data', 'wb') as handle:
                pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # print
            print('\nsaving part '+str(i_data)+'\n')

            # reinit
            L_sdf = []
            L_x_L = []
            L_y_L = []
            L_z_L = []
            L_rbm = []
            i_data = i_data + 1

    else :
        # save counter
        L_counter.append(counter)

        # look for dimensions of box    
        # -x limit
        i_x = 0
        found = False
        while (not found) and (i_x < bin_map.shape[0]):
            if np.max(bin_map[i_x, :, :]) == -1:
                i_x_min_lim = i_x
            else :
                found = True
            i_x = i_x + 1
        if not reduce_sdf:
            i_x_min_lim = 0
        # +x limit
        i_x = bin_map.shape[0]-1
        found = False
        while not found and 0 <= i_x:
            if np.max(bin_map[i_x, :, :]) == -1:
                i_x_max_lim = i_x
            else :
                found = True
            i_x = i_x - 1
        if not reduce_sdf:
            i_x_max_lim = bin_map.shape[0]-1
        # number of nodes on x
        n_nodes_x = i_x_max_lim-i_x_min_lim+1
        # -y limit
        i_y = 0
        found = False
        while (not found) and (i_y < bin_map.shape[1]):
            if np.max(bin_map[:, i_y, :]) == -1:
                i_y_min_lim = i_y
            else :
                found = True
            i_y = i_y + 1
        if not reduce_sdf:
            i_y_min_lim = 0
        # +y limit
        i_y = bin_map.shape[1]-1
        found = False
        while not found and 0 <= i_y:
            if np.max(bin_map[:, i_y, :]) == -1:
                i_y_max_lim = i_y
            else :
                found = True
            i_y = i_y - 1
        if not reduce_sdf:
            i_y_max_lim = bin_map.shape[1]-1
        # number of nodes on y
        n_nodes_y = i_y_max_lim-i_y_min_lim+1
        # -z limit
        i_z = 0
        found = False
        while (not found) and (i_z < bin_map.shape[2]):
            if np.max(bin_map[:, :, i_z]) == -1:
                i_z_min_lim = i_z
            else :
                found = True
            i_z = i_z + 1
        if not reduce_sdf:
            i_z_min_lim = 0
        # +z limit
        i_z = bin_map.shape[2]-1
        found = False
        while not found and 0 <= i_z:
            if np.max(bin_map[:, :, i_z]) == -1:
                i_z_max_lim = i_z
            else :
                found = True
            i_z = i_z - 1
        if not reduce_sdf:
            i_z_max_lim = bin_map.shape[2]-1
        # number of nodes on z
        n_nodes_z = i_z_max_lim-i_z_min_lim+1

        # extraction of data
        bin_map = bin_map[i_x_min_lim:i_x_max_lim+1,
                            i_y_min_lim:i_y_max_lim+1,
                            i_z_min_lim:i_z_max_lim+1] 

        # creation of sub mesh
        m_size = x_L[1] - x_L[0]
        x_L_i = np.arange(-m_size*(n_nodes_x-1)/2,
                            m_size*(n_nodes_x-1)/2+0.1*m_size,
                            m_size)
        y_L_i = np.arange(-m_size*(n_nodes_y-1)/2,
                            m_size*(n_nodes_y-1)/2+0.1*m_size,
                            m_size)
        z_L_i = np.arange(-m_size*(n_nodes_z-1)/2,
                            m_size*(n_nodes_z-1)/2+0.1*m_size,
                            m_size)

        # compute rigid body motion to apply
        rbm = [x_L[i_x_min_lim] - x_L_i[0],
            y_L[i_y_min_lim] - y_L_i[0],
            z_L[i_z_min_lim] - z_L_i[0]]
        
        # compute sdf
        sdf = -skfmm.distance(bin_map, dx=np.array([m_size, m_size, m_size]))
        
        # save 
        L_sdf.append(sdf)
        L_x_L.append(x_L_i)
        L_y_L.append(y_L_i)
        L_z_L.append(z_L_i)
        L_rbm.append(rbm)

        # check the number of grain in the data set
        if len(L_sdf)==n_grain_data or i_sdf==len(L_label)-1:
            # prepare dem_base.py
            dict_save = {
                'L_sdf_i_map': L_sdf,
                'L_x_L': L_x_L,
                'L_y_L': L_y_L,
                'L_z_L': L_z_L,
                'L_rbm': L_rbm
            }
            with open('data/level_set_part'+str(i_data)+'.data', 'wb') as handle:
                pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # print
            print('\nsaving part '+str(i_data)+'\n')

            # reinit
            L_sdf = []
            L_x_L = []
            L_y_L = []
            L_z_L = []
            L_rbm = []
            i_data = i_data + 1

#-------------------------------------------------------------------------------
# prepare dem_base.py
#-------------------------------------------------------------------------------

dict_save = {
    'L_x': x_L,
    'L_y': y_L,
    'L_z': z_L,
    'm_size': m_size,
    'L_pos_w': L_pos_w,
    'n_data_base': i_data - 1
}
with open('data/level_set_part0.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

#-------------------------------------------------------------------------------
# call dem_base.py
#-------------------------------------------------------------------------------

os.system('yadedaily -j 4 -x -n 2_dem_base_3D.py')
#os.system('yadedaily -j 4 -n 2_dem_base_3D.py')

#-------------------------------------------------------------------------------
# read output
#-------------------------------------------------------------------------------

with open('data/output_level_set.data', 'rb') as handle:
    dict_save = pickle.load(handle)
L_surfNodes = dict_save['L_surfNodes']

# plot
fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
ax1.plot(L_surfNodes)
ax1.set_xlabel('grains (-)')
ax1.set_ylabel('number of nodes (-)')
fig.tight_layout()
fig.savefig('2_NodesSurface.png')
plt.close(fig)

# plot
fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
ax1.plot(L_counter)
ax1.set_xlabel('grains (-)')
ax1.set_ylabel('counter (-)')
fig.tight_layout()
fig.savefig('2_Counter.png')
plt.close(fig)

# print
print('n grains', len(L_counter))
print('min counter', min(L_counter))
