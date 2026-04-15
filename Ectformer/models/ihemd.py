import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision

class ReversibleIHEMD:
    def __init__(self, num_ensemble=10, max_imf=10):
        self.num_ensemble = num_ensemble
        self.max_imf = max_imf
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=1)


    def _emd(self, img):
        imfs = []
        residue = img.clone()
        for _ in range(self.max_imf):
            mean_env = self._mean_envelope(residue)
            imf = residue - mean_env
            imfs.append(imf)
            residue = mean_env
        return imfs, residue

    def _mean_envelope(self, img):
        high_pass = self.gaussian_blur(img)
        return img - high_pass

    def ihemd(self, img):
        ensemble_imfs = []
        for _ in range(self.num_ensemble):
            imfs, residue = self._emd(img)
            ensemble_imfs.append((imfs, residue))

        averaged_imfs = [torch.mean(torch.stack([ensemble_imfs[j][0][i] for j in range(self.num_ensemble)]), dim=0) for i in range(self.max_imf)]
        averaged_residue = torch.mean(torch.stack([ensemble_imfs[j][1] for j in range(self.num_ensemble)]), dim=0)

        combined_result = torch.cat([averaged_imfs[0], averaged_imfs[1], averaged_imfs[2], averaged_residue], dim=1)

        return combined_result

    def ihemd_show(self, img):
        ensemble_imfs = []
        for _ in range(self.num_ensemble):
            imfs, residue = self._emd(img)
            ensemble_imfs.append((imfs, residue))

        averaged_imfs = [torch.mean(torch.stack([ensemble_imfs[j][0][i] for j in range(self.num_ensemble)]), dim=0) for i in range(self.max_imf)]
        averaged_residue = torch.mean(torch.stack([ensemble_imfs[j][1] for j in range(self.num_ensemble)]), dim=0)

        combined_result = torch.cat([averaged_imfs[0], averaged_imfs[1], averaged_imfs[2], averaged_residue], dim=1)

        return combined_result, averaged_imfs, averaged_residue


    def reconstruct(self, combined_result):
        split_tensors = torch.split(combined_result, 3, dim=1)

        reconstructed_signal = torch.sum(torch.stack(split_tensors), dim=0)
        return reconstructed_signal

# 示例使用
if __name__ == "__main__":
    # 导入初始图像
    img_path = r''  # 替换为你的图像路径
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 转换为Tensor

    ihemds_processor = ReversibleIHEMD(num_ensemble=3, max_imf=3)
    combined_result, averaged_imfs, averaged_residue = ihemds_processor.ihemd_show(img)

    reconstructed_img = ihemds_processor.reconstruct(combined_result)

    # 可视化初始图像、IMF、残差和重建图像
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes[0, 0].imshow(img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(averaged_imfs[0][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[0, 1].set_title('IMF1')
    axes[1, 0].imshow(averaged_imfs[1][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[1, 0].set_title('IMF2')
    axes[1, 1].imshow(averaged_imfs[2][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[1, 1].set_title('IMF3')
    axes[2, 0].imshow(averaged_residue[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[2, 0].set_title('Residue')
    axes[2, 1].imshow(reconstructed_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[2, 1].set_title('Reconstructed Image')
    plt.show()

