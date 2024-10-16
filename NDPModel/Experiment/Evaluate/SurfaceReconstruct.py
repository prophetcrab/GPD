import os

import numpy as np
import alphashape
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from DataUtils.DataProcessUtils import *

path = r"D:\PythonProject2\GaussianDiffusionFrame\NDPModel\Experiment\Evaluate\DataSet\TrainSet\airplane_0003.txt"
path2 = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples\airplane_new_3.txt"

points = read_pointcould_from_file(path2)
points = normalize(points) * 100



# points = random_sampling_upsample(points, num_new_points=4096)
# points = remove_outliers(points, k=15, threshold=70)

visualize_point_cloud_2(points)

alpha = 0.4
alpha_shape = alphashape.alphashape(points, alpha)


mesh_o3d = alphashape_to_o3d_mesh(alpha_shape)
smoothed_mesh = smooth_mesh(mesh_o3d, iterations=2048)



visualize_trimesh(alpha_shape)

print(alpha_shape)

# save_dir = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples\SamplesGIF"
# save_gif = "airplane12.gif"
#
# save_path = os.path.join(save_dir, save_gif)
#
# visualize_point_cloud_with_rotation_gif(points, save_path=save_path)