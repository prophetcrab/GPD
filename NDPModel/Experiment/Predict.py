import os

from GaussianDiffusion.DDPM import *
from DataUtils.DataProcessUtils import *



if __name__ == '__main__':
    save_path = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples"

    sample_name = "airplane_new_3.txt"

    ddpm = Diffusion()

    sample = ddpm.generate_sample_result(os.path.join(save_path, sample_name))

    sample = normalize(sample)

    print(sample.shape)

    visualize_point_cloud_2(sample)