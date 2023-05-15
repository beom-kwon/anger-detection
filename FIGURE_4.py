import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os


def stride_detector(seq, w):
    right_ankle = seq[:, 42:45]  # Right Ankle
    x_val, y_val, z_val = right_ankle[:, 0], right_ankle[:, 1], right_ankle[:, 2]

    local_maxima_coordinates = []
    for t in range(w, len(z_val) - w):
        if (False not in (z_val[(t - w):t] < z_val[t])) and \
                (False not in (z_val[(t + 1):(t + w)] < z_val[t])):
            local_maxima_coordinates.append(t)

    local_minima_coordinates = []
    for t in range(0, len(local_maxima_coordinates) - 1):
        starting_time = local_maxima_coordinates[t]
        ending_time = local_maxima_coordinates[t + 1]

        local_minima_coordinates.append(starting_time + np.argmin(z_val[starting_time:ending_time]))

    stride_length = []
    for t in range(0, len(local_minima_coordinates) - 1):
        x_old, y_old = x_val[local_minima_coordinates[t]], y_val[local_minima_coordinates[t]]
        x_new, y_new = x_val[local_minima_coordinates[t + 1]], y_val[local_minima_coordinates[t + 1]]

        tmp2 = norm([x_new - x_old, y_new - y_old])
        stride_length.append(tmp2)

    return np.array(stride_length)


def file_read(full_dir):
    ptd_file = open(full_dir, 'r')
    seq = []
    while True:
        data = ptd_file.readline()
        row = data.split(' ')
        if len(row) == 46:
            row.remove('\n')
            row_f = []
            for str_element in row:
                float_element = float(str_element)
                row_f.append(float_element)

            seq.append(row_f)

        if not data:
            break

    ptd_file.close()
    return seq


person_id = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun',
             'ele', 'emm', 'gra', 'ian', 'jan', 'jen', 'jua', 'kat', 'lin', 'mac',
             'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi', 'ste', 'vas']

common_dir = os.getcwd() + '\\bml\\'

# Control Parameters
window_size = 30      # Window Size used in the function 'stride_detector'
split_indices = [1]
print('** Window size: ', window_size)

idx = 0  # 0~29
pid = person_id[idx]
pt_file_list = os.listdir(common_dir + pid + '_pt')

for file in pt_file_list:
    if file == 'chr_walk_nu_1_fin.ptd':
        # In the ptd file above, there are only (static) pose data rather than gait data.
        # For this reason, we ignore this file.
        continue

    pt_file_full_dir = common_dir + pid + '_pt\\' + file
    seq_list = file_read(pt_file_full_dir)

    seq_list = np.array(seq_list)
    print(seq_list.shape)
    if '_an_' in file:
        break

plt.figure()
plt.plot(seq_list[:, 44], c='black')
plt.grid(True)
plt.xlabel('Frame')
plt.ylabel('Height of the Right Ankle')
plt.show()
