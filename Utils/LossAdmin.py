import os
import matplotlib.pyplot as plt

class LossAdmin:

    def __init__(self, loss_file_output_path, loss_img_output_path):
        """
        :param loss_file_output_path: Loss数据储存的目标文件
        :param loss_img_output_path: Loss折线图储存的目标文件夹
        :param start_epoch: 本次开始训练的epoch
        """
        self.loss_file_output_path = loss_file_output_path
        self.loss_img_output_path = loss_img_output_path
        #self.start_epoch = start_epoch
    def write_loss(self, epoch, loss):
        """检查文件是否存在"""
        file_exists = os.path.exists(self.loss_file_output_path)

        with open(self.loss_file_output_path, 'a+') as file:

            if not file_exists:
                file.write("Epoch,Loss\n")
            else:
                file.seek(0)
                first_line = file.readline().strip()
                if first_line != "Epoch,Loss":
                    file.write("Epoch,Loss\n")

            file.write(f"{epoch},{loss}\n")

    def plot_loss(self):
        epochs = []
        losses = []

        if not os.path.exists(self.loss_file_output_path):
            ValueError("Loss file is not exist")
            return

        with open(self.loss_file_output_path, 'r') as file:
            lines = file.readlines()

            for line in lines[1:]:

                if not line or ',' not in line:  # 检查是否为空行或是否有逗号
                    print(f"跳过无效行: {line}")
                    continue

                try:
                    epoch, loss = line.split(',')
                    epochs.append(int(epoch))
                    losses.append(float(loss))
                except ValueError:
                    print(f"解析失败的行: {line}")
                    continue

        if not epochs or not losses:
            ValueError("Data Error")
            return

        plt.figure()
        plt.plot(epochs, losses, marker='o', markersize='0.05', linestyle='-', color='blue', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.grid(True)
        plt.legend()

        if not os.path.exists(self.loss_img_output_path):
            os.makedirs(self.loss_img_output_path)

        output_file = os.path.join(self.loss_img_output_path, 'loss.png')
        plt.savefig(output_file)
        plt.close()

        print("Save successful")





if __name__ == '__main__':
    file_path = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\output.txt"
    img_path = r"D:\PythonProject2\GaussianDiffusionFrame\Logs"
    LossAdmin = LossAdmin(file_path,img_path,0)
    LossAdmin.plot_loss()
