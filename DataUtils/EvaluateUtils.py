import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from DataUtils.DataProcessUtils import *



def chamfer_distance(source, target):
    """
    计算两个点云的 Chamfer Distance。
    """
    kdtree_source = cKDTree(source)
    kdtree_target = cKDTree(target)
    dist_source_to_target, _ = kdtree_source.query(target, k=1)
    dist_target_to_source, _ = kdtree_target.query(source, k=1)
    cd = np.mean(dist_source_to_target ** 2) + np.mean(dist_target_to_source ** 2)
    return cd


def emd_distance(source, target):
    """
    计算两个点云的 Earth Mover's Distance (EMD)。
    """
    # 计算两个点云之间的距离矩阵
    distance_matrix = np.linalg.norm(source[:, np.newaxis] - target, axis=2)

    # 使用线性分配算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # 计算匹配后的总距离
    emd = distance_matrix[row_ind, col_ind].sum() / len(row_ind)  # 平均 EMD 距离
    return emd


def compute_mmd(point_clouds_real, point_clouds_gen, distance_func):
    """
    计算 MMD (Maximum Mean Discrepancy)。

    参数:
    - point_clouds_real: List of np.array, 真实点云列表，每个点云为 (n, 3) 数组。
    - point_clouds_gen: List of np.array, 生成点云列表，每个点云为 (n, 3) 数组。
    - distance_func: function, 用于计算点云之间距离的函数，如 Chamfer Distance 或 EMD。

    返回:
    - mmd_value: float, 计算的 MMD 值。
    """
    n_real = len(point_clouds_real)
    n_gen = len(point_clouds_gen)

    # 计算生成点云和真实点云之间的距离矩阵
    total_distance = 0.0
    for point_cloud_gen in point_clouds_gen:
        dist_to_real = []
        for point_cloud_real in point_clouds_real:

            #dist_to_real = distance_func(point_cloud_real, point_cloud_gen)
            point_cloud_gen, _, _ = align_point_clouds(point_cloud_gen, point_cloud_real)
            point_cloud_gen = center_and_scale_point_cloud(point_cloud_gen)
            point_cloud_gen = normalize(point_cloud_gen) / 10
            point_cloud_real = normalize(point_cloud_real) / 10
            dist_to_real.append(distance_func(point_cloud_real, point_cloud_gen))

        total_distance += np.min(dist_to_real)  # 取每个生成点云到真实点云的最小距离
        # print(total_distance)
    # 取平均
    mmd_value = total_distance / n_gen
    return mmd_value


def compute_cov(point_clouds_real, point_clouds_gen, distance_func, threshold=1e-5):
    """
    计算 COV (Coverage)。

    参数:
    - point_clouds_real: List of np.array, 真实点云列表，每个点云为 (n, 3) 数组。
    - point_clouds_gen: List of np.array, 生成点云列表，每个点云为 (n, 3) 数组。
    - distance_func: function, 用于计算点云之间距离的函数，如 Chamfer Distance 或 EMD。
    - threshold: float, 覆盖的阈值，判断点云是否足够接近。

    返回:
    - cov_value: float, 计算的 COV 值，百分比。
    """
    n_real = len(point_clouds_real)

    covered_real_clouds = 0

    dist_to_gen = distance_func(point_clouds_real, point_clouds_gen)
    if np.min(dist_to_gen) < threshold:  # 判断是否覆盖
        covered_real_clouds += 1

    # 计算覆盖度百分比
    cov_value = (covered_real_clouds / n_real) * 100
    return cov_value