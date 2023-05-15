import matplotlib.pyplot as plt
import numpy as np
import sys
import os

person_id = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun',
             'ele', 'emm', 'gra', 'ian', 'jan', 'jen', 'jua', 'kat', 'lin', 'mac',
             'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi', 'ste', 'vas']

common_dir = os.getcwd() + '\\bml\\'
for pid in person_id:
    pt_file_list = os.listdir(common_dir + pid + '_pt')
    for file in pt_file_list:
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
        ndy_arr = np.array(seq_list)
        # print(ndy_arr.shape)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        offset = 5
        x_max = np.max(ndy_arr[:, 0::3])
        x_min = np.min(ndy_arr[:, 0::3])
        y_max = np.max(ndy_arr[:, 1::3])
        y_min = np.min(ndy_arr[:, 1::3])
        z_max = np.max(ndy_arr[:, 2::3])
        z_min = np.min(ndy_arr[:, 2::3])

        for frm in range(0, ndy_arr.shape[0]):
            x = ndy_arr[frm, 0::3]
            y = ndy_arr[frm, 1::3]
            z = ndy_arr[frm, 2::3]

            ax.set_xlim3d([x_min - offset, x_max + offset])
            ax.set_ylim3d([y_min - offset, y_max + offset])
            ax.set_zlim3d([z_min - offset, z_max + offset])

            ax.set_box_aspect((1, 1, 1))

            # Creating plot
            ax.scatter3D(x, y, z, color='green')
            # Head - Neck
            ax.plot3D([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], 'gray')
            # Neck - Pelvis
            ax.plot3D([x[1], x[8]], [y[1], y[8]], [z[1], z[8]], 'gray')
            # Left Arm
            ax.plot3D([x[1], x[2]], [y[1], y[2]], [z[1], z[2]], 'blue')
            ax.plot3D([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], 'blue')
            ax.plot3D([x[3], x[4]], [y[3], y[4]], [z[3], z[4]], 'blue')
            # Right Arm
            ax.plot3D([x[1], x[5]], [y[1], y[5]], [z[1], z[5]], 'red')
            ax.plot3D([x[5], x[6]], [y[5], y[6]], [z[5], z[6]], 'red')
            ax.plot3D([x[6], x[7]], [y[6], y[7]], [z[6], z[7]], 'red')
            # Left Leg
            ax.plot3D([x[8], x[9]], [y[8], y[9]], [z[8], z[9]], 'blue')
            ax.plot3D([x[9], x[10]], [y[9], y[10]], [z[9], z[10]], 'blue')
            ax.plot3D([x[10], x[11]], [y[10], y[11]], [z[10], z[11]], 'blue')
            # Right Leg
            ax.plot3D([x[8], x[12]], [y[8], y[12]], [z[8], z[12]], 'red')
            ax.plot3D([x[12], x[13]], [y[12], y[13]], [z[12], z[13]], 'red')
            ax.plot3D([x[13], x[14]], [y[13], y[14]], [z[13], z[14]], 'red')

            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(0.01)
            ax.clear()

        sys.exit()
