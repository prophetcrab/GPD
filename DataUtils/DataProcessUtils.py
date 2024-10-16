import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from PIL import Image




def visualize_point_cloud(point_cloud, color=(0, 0, 255), size=1):
    """
    :param point_cloud: [N, C]
    :param color:
    :param size:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color(color)

    #axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=(0, 0, 0))

    o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud_2(points, point_size=5,  color=None):
    """
    使用open3d可视化点云，并调整点的大小和颜色。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)。
        point_size (int): 点的大小。
        color (tuple or np.ndarray): 点的颜色 (R, G, B)，每个通道的值在[0, 1]之间。
    """
    # 创建open3d的点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果传入了颜色，将颜色赋值给点云
    if color is not None:
        if isinstance(color, tuple):
            # 使用相同的颜色给所有点
            pcd.paint_uniform_color(color)
        elif isinstance(color, np.ndarray) and color.shape[0] == points.shape[0]:
            # 使用每个点的颜色
            pcd.colors = o3d.utility.Vector3dVector(color)

    # 创建可视化器并调整点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # 调整点的大小
    render_option.background_color = np.array([0, 0, 0])  # 设置背景颜色为黑色
    render_option.show_coordinate_frame = True  # 显示坐标系

    # 运行可视化
    vis.run()
    vis.destroy_window()


def visualize_point_cloud_with_plt(point_cloud):
    """
        使用 matplotlib 可视化三维点云

        参数:
        - point_cloud: numpy 数组，形状为 (N, 3)，表示点云数据
        """
    # 创建一个新的图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', s=1)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()


"""可视化ttrimesh对象"""
def visualize_trimesh(mesh):
    """
    使用 open3d 可视化 trimesh 对象
    :param mesh: trimesh.Trimesh 对象
    """
    # 提取三角网格的顶点和面
    # 提取三角网格的顶点和面
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # 创建Open3D三角网格对象
    mesh_o3d = o3d.geometry.TriangleMesh()

    # 设置顶点和三角形面
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    # 计算法线以便更好的渲染效果
    mesh_o3d.compute_vertex_normals()

    # 创建颜色数组，设置每个顶点的颜色为灰色（不透明）
    # 如果需要透明效果，可以将最后一维的alpha值调小
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * len(vertices))  # 灰色

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加网格到窗口
    vis.add_geometry(mesh_o3d)

    # 获取渲染选项并设置其他可视化属性
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True  # 显示背面
    render_option.background_color = np.array([1, 1, 1])  # 白色背景

    # 启动可视化
    vis.run()
    vis.destroy_window()

def visualize_o3d_mesh(o3dmesh):
    o3d.visualization.draw_geometries([o3dmesh])



def visualize_point_cloud_with_rotation_gif(points, point_size=5, color=None, save_path="rotation_animation.gif"):
    """
    可视化点云，并保存360度旋转动画为GIF。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)。
        point_size (int): 点的大小。
        color (tuple or np.ndarray): 点的颜色 (R, G, B)，每个通道的值在[0, 1]之间。
        save_path (str): 保存动画的路径。
    """
    # 创建Open3D的点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果传入了颜色，将颜色赋给点云
    if color is not None:
        if isinstance(color, tuple):
            pcd.paint_uniform_color(color)
        elif isinstance(color, np.ndarray) and color.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(color)

    # 创建可视化器并调整点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0, 0, 0])
    render_option.show_coordinate_frame = True

    # 保存每一帧的图像
    images = []
    angle_step = 30  # 每次旋转的角度
    for angle in range(0, 3600, angle_step):
        ctr = vis.get_view_control()
        ctr.rotate(angle_step, 0)  # 水平旋转
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)  # 转换为uint8
        pil_image = Image.fromarray(image)
        images.append(pil_image)



    # 销毁窗口
    vis.destroy_window()

    # 使用PIL保存为GIF动图
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0)


def read_pointcould_from_file(filepath):
    with open(filepath, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
        data = data[:, :3]
        return data

def voxel_downsample_point_cloud(point_clouds, voxel_size=0.05, target_num_points=2048):
    """
    对形状为 (B, C, N) 的批次点云进行体素下采样，并调整点数到目标大小 N。

    参数:
    - point_clouds: numpy 数组，形状为 (B, C, N)，表示 B 个点云批次，每个点云有 N 个点，每个点有 C 个通道。
    - voxel_size: 体素大小，控制下采样的粒度，默认值为 0.05。
    - target_num_points: 目标点数 N，下采样后的每个点云将调整到此数量。

    返回:
    - downsampled_batch: 下采样并调整后的点云数据，形状为 (B, C, target_num_points)。
    """
    B, C, N = point_clouds.shape
    downsampled_result = np.zeros((B, C, target_num_points))

    for i in range(B):
        # 将单个点云数据提取出来，并转换为 (N, 3) 形式
        point_cloud = point_clouds[i].T  # 转置为 (N, C)

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # 执行体素下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # 获取下采样后的点并转换为 numpy 数组
        downsampled_points = np.asarray(downsampled_pcd.points).T  # 转置回 (C, M) 形式
        M = downsampled_points.shape[1]

        # 调整点数到目标大小
        if M >= target_num_points:
            # 如果下采样后的点数大于目标点数，随机选择 target_num_points 个点
            selected_indices = np.random.choice(M, target_num_points, replace=False)
            downsampled_points = downsampled_points[:, selected_indices]
        else:
            # 如果下采样后的点数少于目标点数，随机重复点来填充
            repeated_indices = np.random.choice(M, target_num_points - M, replace=True)
            downsampled_points = np.concatenate([downsampled_points, downsampled_points[:, repeated_indices]], axis=1)

        # 将处理后的点云添加到结果数组中
        downsampled_result[i] = downsampled_points

    return downsampled_result


def normalize(data):
    min_coords = np.min(data, axis=0)
    max_coords = np.max(data, axis=0)

    range_coords = max_coords - min_coords

    max_range = np.max(range_coords)

    normalized_data = (data - min_coords) / max_range
    return normalized_data


def normalize_to_minus_one_one(data):
    """
    将点云数据归一化到 [-1, 1] 范围，并保持形状不变。

    参数:
    - data: np.array, 点云数据，形状为 (n, 3) 或 (n, m) 的数组。

    返回:
    - normalized_data: np.array, 归一化到 [-1, 1] 的点云数据。
    """
    # 计算每个维度的最小和最大值
    min_coords = np.min(data, axis=0)
    max_coords = np.max(data, axis=0)

    # 计算每个维度的范围
    range_coords = max_coords - min_coords

    # 确保没有除以零的情况，避免数值错误
    range_coords[range_coords == 0] = 1

    # 将数据归一化到 [0, 1] 的范围
    normalized_data = (data - min_coords) / range_coords

    # 将归一化数据从 [0, 1] 线性变换到 [-1, 1]
    normalized_data = normalized_data * 2 - 1

    return normalized_data

def voxel_downsampling(data, voxel_size):

    min_coords = np.min(data, axis=0)
    voxel_indices = np.floor((data - min_coords) / voxel_size).astype(int)
    voxel_grid = {}

    for point, voxel_idx in zip(data, voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_grid:
            voxel_grid[voxel_key] = []
        voxel_grid[voxel_key].append(point)

    downsampled_point_cloud = np.array([np.mean(points, axis=0) for points in voxel_grid.values()])


    return downsampled_point_cloud

def random_sampling(data, target_num_points):
    if target_num_points >= data.shape[0]:
        return data
    selected_indices = np.random.choice(data.shape[0], target_num_points, replace=False)
    downsampled_point_cloud = data[selected_indices]
    return downsampled_point_cloud


def point_cloud_upsample(points, upsample_factor=2, radius=0.1):
    """
    改进的点云上采样，通过控制插值范围和距离。

    参数:
        points (np.ndarray): 输入点云，形状为 (N, 3)。
        upsample_factor (int): 上采样因子。
        radius (float): 限制生成新点的最大插值半径，避免插值时生成过远的点。

    返回:
        np.ndarray: 上采样后的点云。
    """
    # 创建一个KDTree，用于找到最近邻
    tree = cKDTree(points)

    # 原始点数
    N = points.shape[0]

    new_points = [points]

    # 遍历每个点并找到其邻居
    for i in range(N):
        distances, idx = tree.query(points[i], k=upsample_factor + 1)  # 找到k+1个最近邻（包括自己）

        # 对于每个最近邻（排除自己），在规定半径内插值生成新点
        for j in range(1, upsample_factor + 1):
            if distances[j] <= radius:  # 限制在radius半径内的插值
                new_point = (points[i] + points[idx[j]]) / 2.0  # 简单平均生成新点
                new_points.append(new_point)

    # 将生成的点云转换为numpy数组
    upsampled_points = np.vstack(new_points)

    return upsampled_points


def random_sampling_upsample(points, num_new_points):
    """
    通过随机选择两个点并插值生成新点的方式上采样点云。

    参数:
        points (np.ndarray): 输入点云，形状为 (N, 3)。
        num_new_points (int): 需要生成的新点的数量。

    返回:
        np.ndarray: 包含新点的上采样点云。
    """
    N = points.shape[0]
    new_points = []

    for _ in range(num_new_points):
        idx1, idx2 = np.random.choice(N, 2, replace=False)  # 随机选择两个不同的点
        t = np.random.rand()  # 随机生成插值系数
        new_point = (1 - t) * points[idx1] + t * points[idx2]  # 插值生成新点
        new_points.append(new_point)

    # 将原始点和新生成的点合并
    upsampled_points = np.vstack([points, np.array(new_points)])

    return upsampled_points


def surface_normal_upsample(points, num_new_points=100, radius=0.1):
    """
    基于法线插值的上采样方法。

    参数:
        points (np.ndarray): 输入点云，形状为 (N, 3)。
        num_new_points (int): 新生成的点数量。
        radius (float): 用于计算法线的邻域半径。

    返回:
        np.ndarray: 上采样后的点云。
    """
    # 使用最近邻搜索来估计法线
    nbrs = NearestNeighbors(n_neighbors=10, radius=radius).fit(points)
    _, indices = nbrs.kneighbors(points)

    new_points = []
    for i in range(num_new_points):
        idx = np.random.randint(0, points.shape[0])
        neighbors = points[indices[idx]]

        # 计算法线
        covariance_matrix = np.cov(neighbors.T)
        _, _, normal = np.linalg.svd(covariance_matrix)

        # 沿着法线方向生成一个新点
        displacement = np.random.normal(scale=0.01, size=3)
        new_point = points[idx] + displacement * normal[-1]
        new_points.append(new_point)

    return np.vstack([points, np.array(new_points)])


def poisson_upsample(points, depth=8):
    """
    使用Poisson重建算法对点云进行上采样。

    参数:
        points (np.ndarray): 输入点云，形状为 (N, 3)。
        depth (int): 控制细节层次，值越大重建的细节越多。

    返回:
        np.ndarray: 上采样后的点云。
    """
    # 转换为open3d点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 估计法线
    pcd.estimate_normals()

    # Poisson重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 提取上采样后的点
    upsampled_points = np.asarray(mesh.vertices)

    return upsampled_points

def remove_outliers(points, k=10, threshold=1.0):
    """
    移除点云中的离群点。

    参数:
        points (np.ndarray): 输入点云，形状为 (N, 3)。
        k (int): 最近邻点的数量，用于计算每个点的平均距离，默认值为10。
        threshold (float): 离群点的距离阈值，平均距离超过该值的点会被视为离群点。

    返回:
        np.ndarray: 去除离群点后的点云。
    """
    # 使用KDTree加速最近邻搜索
    tree = cKDTree(points)

    # 计算每个点到其 k 个最近邻的平均距离
    distances, _ = tree.query(points, k=k + 1)  # 包括自己，所以使用k+1
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身的距离（即第一个最近邻是自己）

    # 根据阈值判断离群点
    non_outliers_mask = avg_distances < threshold

    # 过滤掉离群点
    filtered_points = points[non_outliers_mask]

    return filtered_points

# 1. 点云去噪
def remove_noise(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    使用统计滤波法去除点云中的噪声
    :param pcd: Open3D 点云对象
    :param nb_neighbors: 用于估算每个点领域的点数
    :param std_ratio: 离群值标准差比率
    :return: 去噪后的点云
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_inlier = pcd.select_by_index(ind)
    return pcd_inlier




#点云对齐
def align_point_clouds(source, target):
    """
    对两个形状为 (n, 3) 的点云进行对齐，使 source 点云对齐到 target 点云。
    使用 Kabsch 算法进行旋转和平移的估计。

    参数:
    - source: np.array, (n, 3)，待对齐的点云。
    - target: np.array, (n, 3)，参考点云。

    返回:
    - aligned_source: np.array, (n, 3)，对齐后的点云。
    - R: np.array, (3, 3)，旋转矩阵。
    - t: np.array, (3,)，平移向量。
    """

    assert source.shape == target.shape, "两个点云必须具有相同的形状"

    # 计算质心
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # 去质心化
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # 计算协方差矩阵
    H = np.dot(source_centered.T, target_centered)

    # 使用奇异值分解（SVD）计算旋转矩阵
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 确保旋转矩阵为右手系（防止反射问题）
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移向量
    t = centroid_target - np.dot(centroid_source, R)

    # 应用旋转和平移到源点云
    aligned_source = np.dot(source, R) + t

    return aligned_source, R, t


# 将 Alpha Shape 转换为 Open3D 的 TriangleMesh
def alphashape_to_o3d_mesh(alpha_shape):
    vertices = np.array(alpha_shape.vertices)
    faces = np.array(alpha_shape.faces)

    # 创建 Open3D 三角网格对象
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    # 计算法线
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d



def estimate_normals(pcd):
    """
    估计点云的法线，供 Ball Pivoting 重建使用
    :param pcd: Open3D 点云对象
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)  # 确保法线方向一致

"""BPA重建算法"""
def ball_pivoting_mesh_reconstruction(pcd):
    """
    使用 Ball Pivoting 算法对点云进行网格重建
    :param pcd: Open3D 点云对象
    :return: 重建的三角网格
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    estimate_normals(pcd_o3d)
    radii = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.1, 0.2, 1]  # 设置不同的半径，以适应点云密度
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_o3d, o3d.utility.DoubleVector(radii))
    return mesh

"""MLS重建算法"""
def mls_smooth_reconstruction(pcd):
    """
    使用 MLS 算法对点云进行平滑处理并重建曲面
    :param pcd: np点云
    :return: 重建的三角网格
    """
    # MLS 平滑
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_mls = pcd_o3d.voxel_down_sample(voxel_size=0.2)
    pcd_mls.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_mls = pcd_mls.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

    # Poisson 重建
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_mls, depth=8)

    return mesh

# 对网格进行平滑处理
def smooth_mesh(mesh_o3d, iterations=10):
    """
    使用 Open3D 对网格进行平滑处理
    :param mesh_o3d: open3d.geometry.TriangleMesh 对象
    :param iterations: 平滑处理的迭代次数
    :return: 平滑后的三角网格
    """
    # 使用 laplacian 平滑网格
    mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=iterations)
    return mesh_o3d


def export_point_cloud_to_ply(points, filename, color=(1, 0.7, 0.2)):
    """
    将点云导出为PLY文件，便于在Blender中打开。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)。
        filename (str): 要保存的PLY文件名称。
        color (tuple): 点云的颜色 (R, G, B)，每个通道的值在[0, 1]之间。
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 为点云设置颜色
    colors = np.tile(np.array(color), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存点云为PLY文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存为 {filename}")

"中心化"
def center_and_scale_point_cloud(point_cloud, scale_factor=0.5):
    """
    将点云向中心（质心）靠拢，但不改变形状。

    参数:
    - point_cloud: NumPy 数组，形状 (N, 3)，表示点云
    - scale_factor: float，缩放因子，1.0 表示不缩放，0.5 表示点云向质心靠拢 50%

    返回:
    - 缩放后的点云
    """
    # 计算点云的质心
    centroid = np.mean(point_cloud, axis=0)

    # 将每个点相对于质心的位置进行缩放
    scaled_point_cloud = centroid + (point_cloud - centroid) * scale_factor

    return scaled_point_cloud


if __name__ == '__main__':
    B, C, N = 128, 3, 5000
    point_cloud = np.random.rand(B, C, N)

    voxel_size = 0.05
    downsampled_point_cloud = voxel_downsample_point_cloud(point_cloud)
    #downsampled_point_cloud = downsample_point_cloud(downsampled_point_cloud)

    print(downsampled_point_cloud.shape)


