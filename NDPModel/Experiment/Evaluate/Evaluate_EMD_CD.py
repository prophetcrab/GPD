from DataUtils.DataProcessUtils import *
from DataUtils.EvaluateUtils import *

import os

def load_data(folder_path, label, repeat_times=1):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            points = np.loadtxt(file_path, delimiter=',')
            if label == 1:
                points = voxel_downsampling(points, voxel_size=0.001)
                points = random_sampling(points, target_num_points=2048)
            # 重复数据 repeat_times 次
            for _ in range(repeat_times):
                data.append(points[:, :3])  # 只取前三列（真实数据只保留前三位）
    return np.array(data)


if __name__ == '__main__':
    point_cloud_gen = read_pointcould_from_file(r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples\airplane_new_1.txt")
    point_clouds_real = load_data(r"D:\PythonProject2\GaussianDiffusionFrame\NDPModel\Experiment\Evaluate\DataSet\RealData", label=1)
    point_clouds_gen = load_data(r"D:\PythonProject2\GaussianDiffusionFrame\NDPModel\Experiment\Evaluate\DataSet\RealData", label=1)

    print(point_clouds_gen.shape)
    print(point_clouds_real.shape)


    # mmd_cd = compute_mmd(point_clouds_real, point_clouds_gen, chamfer_distance)
    #
    # # mmd_emd = compute_mmd(point_clouds_real, point_clouds_gen, emd_distance)
    # print("MMD_CD:", mmd_cd)
    # print("MMD_EMDl", mmd_emd)

    point_cloud_gen = normalize(point_cloud_gen)



    visualize_point_cloud_2(point_cloud_gen)


    emd = []
    for point_cloud in point_clouds_real:
        point_cloud = normalize(point_cloud)
        point_cloud_gen, _, _ = align_point_clouds(point_cloud_gen, point_cloud)
        point_cloud_gen = point_cloud_gen
        emd.append(emd_distance(point_cloud, point_cloud_gen))


    emd_min = np.min(emd)


    cd = []
    for point_cloud in point_clouds_real:
        point_cloud = normalize(point_cloud)
        point_cloud_gen, _, _ = align_point_clouds(point_cloud_gen, point_cloud)
        point_cloud_gen = point_cloud_gen
        cd.append(chamfer_distance(point_cloud, point_cloud_gen))

    chamfer_min = np.min(cd)


    print("CD:", chamfer_min )
    print("EMD:", emd_min )

