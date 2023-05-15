import numpy as np
from numpy.linalg import norm
import os

# Joint Index Information
# 00: Head
# 01: Neck
# 02: Left Shoulder
# 03: Left Elbow
# 04: Left Wrist
# 05: Right Shoulder
# 06: Right Elbow
# 07: Right Wrist
# 08: Pelvis
# 09: Left Hip
# 10: Left Knee
# 11: Left Ankle
# 12: Right Hip
# 13: Right Knee
# 14: Right Ankle


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
    cadence = []
    velocity = []
    for t in range(0, len(local_minima_coordinates) - 1):
        tmp1 = local_minima_coordinates[t + 1] - local_minima_coordinates[t]
        cadence.append(tmp1)

        x_old, y_old = x_val[local_minima_coordinates[t]], y_val[local_minima_coordinates[t]]
        x_new, y_new = x_val[local_minima_coordinates[t + 1]], y_val[local_minima_coordinates[t + 1]]

        tmp2 = norm([x_new - x_old, y_new - y_old])
        stride_length.append(tmp2)

        if tmp1 != 0:
            velocity.append(tmp2 / tmp1)

    if np.array(stride_length).size == 0:
        stride_length = 0
    if np.array(cadence).size == 0:
        cadence = 0
    if np.array(velocity).size == 0:
        velocity = 0

    return np.array(stride_length), np.array(cadence), np.array(velocity)


def neck_angle(sample):
    angle = []
    for row1 in sample:
        # head (0), neck (1), pelvis (8), left_hip (9), right_hip (12)
        head, neck, pelvis, left_hip, right_hip = row1[0:3], row1[3:6], row1[24:27], row1[27:30], row1[36:39]

        x0, y0 = head[0], head[1]
        x1, y1 = neck[0], neck[1]
        x8, y8 = pelvis[0], pelvis[1]
        x9, y9 = left_hip[0], left_hip[1]
        x12, y12 = right_hip[0], right_hip[1]

        k0 = ((x8 - x0) * (x12 - x9) + (y8 - y0) * (y12 - y9)) / ((x12 - x9)**2 + (y12 - y9)**2)
        x0_prime = k0 * (x12 - x9) + x0
        y0_prime = k0 * (y12 - y9) + y0

        k1 = ((x8 - x1) * (x12 - x9) + (y8 - y1) * (y12 - y9)) / ((x12 - x9) ** 2 + (y12 - y9) ** 2)
        x1_prime = k1 * (x12 - x9) + x1
        y1_prime = k1 * (y12 - y9) + y1

        head_prime = np.array([x0_prime, y0_prime, head[2]])
        neck_prime = np.array([x1_prime, y1_prime, neck[2]])

        head_double_prime = np.array([x0_prime, y0_prime, 0])  # Foot of Perpendicular
        neck_double_prime = np.array([x1_prime, y1_prime, 0])  # Foot of Perpendicular
        tmp = np.arcsin(norm(head_double_prime - neck_double_prime) / norm(head_prime - neck_prime))
        if not np.isnan(tmp):
            angle.append(tmp)

    return np.array(angle)


def thorax_angle(sample):
    angle = []
    for row2 in sample:
        # neck (1), pelvis (8), left_hip (9), right_hip (12)
        neck, pelvis, left_hip, right_hip = row2[3:6], row2[24:27], row2[27:30], row2[36:39]

        x1, y1 = neck[0], neck[1]
        x8, y8 = pelvis[0], pelvis[1]
        x9, y9 = left_hip[0], left_hip[1]
        x12, y12 = right_hip[0], right_hip[1]

        k1 = ((x8 - x1) * (x12 - x9) + (y8 - y1) * (y12 - y9)) / ((x12 - x9) ** 2 + (y12 - y9) ** 2)
        x1_prime = k1 * (x12 - x9) + x1
        y1_prime = k1 * (y12 - y9) + y1

        neck_prime = np.array([x1_prime, y1_prime, neck[2]])  # Foot of Perpendicular
        neck_double_prime = np.array([x1_prime, y1_prime, 0])  # Foot of Perpendicular
        pelvis_double_prime = np.array([pelvis[0], pelvis[1], 0])  # Foot of Perpendicular

        tmp = np.arcsin(norm(neck_double_prime - pelvis_double_prime) / norm(neck_prime - np.array(pelvis)))
        if not np.isnan(tmp):
            angle.append(tmp)

    return np.array(angle)


person_id = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun',
             'ele', 'emm', 'gra', 'ian', 'jan', 'jen', 'jua', 'kat', 'lin', 'mac',
             'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi', 'ste', 'vas']

common_dir = os.getcwd() + '\\bml\\'

# Control Parameters
window_size = 30              # This parameter is used in the function 'stride_detector'
# split_indices = [1]           # 'feature1.npz' is generated (<- conventional feature extraction technique)
# split_indices = [1, 2]        # 'feature3.npz' is generated
# split_indices = [2, 3]        # 'feature5.npz' is generated
# split_indices = [1, 2, 4]     # 'feature7.npz' is generated
# split_indices = [1, 3, 5]     # 'feature8.npz' is generated
split_indices = [1, 2, 3, 5]  # 'feature11.npz' is generated
# sum(split_indices) is equal to the number of classifiers used in ensemble model
# see Table 1 in the paper.

print('** Window size: ', window_size)

feature_container = []
label_container = []  # 1: angry, 0: the others
for idx in range(len(person_id)):
    print(str(idx + 1) + '/' + str(len(person_id)))
    pid = person_id[idx]
    pt_file_list = os.listdir(common_dir + pid + '_pt')
    for file in pt_file_list:
        if file == 'chr_walk_nu_1_fin.ptd':
            # In the ptd file above, there are only (static) pose data rather than gait data.
            # For this reason, we ignore this file.
            continue

        if '_an_' in file:
            label_container.append(1)
        elif '_ha_' in file:
            label_container.append(0)
        elif '_nu_' in file:
            label_container.append(0)
        elif '_sa_' in file:
            label_container.append(0)

        pt_file_full_dir = common_dir + pid + '_pt\\' + file
        ptd_file = open(pt_file_full_dir, 'r')
        seq_list = []
        while True:
            data = ptd_file.readline()
            row = data.split(' ')
            if len(row) == 46:
                row.remove('\n')
                row_f = []
                for str_element in row:
                    float_element = float(str_element)
                    row_f.append(float_element)

                seq_list.append(row_f)

            if not data:
                break

        ptd_file.close()

        # Feature Extraction
        # LOS: Length of One Stride
        # TOS: Time for One Stride
        # VEL: Velocity (= LOS / TOS)
        # NEC: Neck Angle
        # THO: Thorax Angle

        feature_list = []
        seq_arr = np.array(seq_list, dtype=np.float32)
        for i1 in split_indices:
            seq_part = np.array_split(seq_arr, i1)
            for i2 in range(i1):
                LOS, TOS, VEL = stride_detector(np.array(seq_part[i2], dtype=np.float32), window_size)
                NEC = neck_angle(np.array(seq_part[i2], dtype=np.float32))
                THO = thorax_angle(np.array(seq_part[i2], dtype=np.float32))

                feature_list.append([np.min(LOS), np.mean(LOS), np.max(LOS),
                                     np.min(TOS), np.mean(TOS), np.max(TOS),
                                     np.min(VEL), np.mean(VEL), np.max(VEL),
                                     np.min(NEC), np.mean(NEC), np.max(NEC),
                                     np.min(THO), np.mean(THO), np.max(THO)])

        feature_arr = np.array(feature_list)
        feature_arr = feature_arr.reshape(feature_arr.shape[0] * feature_arr.shape[1])

        feature_container.append(feature_arr)

feature_container = np.array(feature_container)
label_container = np.array(label_container, dtype=np.int8)
print(feature_container.shape, label_container.shape)
name = 'feature' + str(sum(split_indices)) + '.npz'
np.savez(name, x=feature_container, y=label_container)
