import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import config as c
import datasets
import timm.scheduler
import cv2
from models.ihemd import ReversibleIHEMD
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from models.ectformer import Ectformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img_1 = np.array(img1).astype(np.float64)
    img_2 = np.array(img2).astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img_1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img_2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img_1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img_2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img_1 * img_2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def computeSSIM(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1_in = np.transpose(img1, (1, 2, 0))
    img2_in = np.transpose(img2, (1, 2, 0))
    if not img1_in.shape == img2_in.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1_in.ndim == 2:
        return ssim(img1_in, img2_in)
    elif img1_in.ndim == 3:
        # 多通道
        if img1_in.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1_in, img2_in))
            return np.array(ssims).mean()
        # 单通道
        elif img1_in.shape[2] == 1:
            return ssim(np.squeeze(img1_in), np.squeeze(img2_in))
    else:
        raise ValueError('Wrong input image dimensions.')



def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)

    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def computermse(test1, test2):

    rmse = np.sqrt(np.mean((test1-test2)**2))
    return rmse


def computemae(test1, test2):

    mae = np.mean(np.abs(test1-test2))
    return mae

def calculate_ms_ssim(image1, image2):

    # 计算 MS-SSIM，值越接近 1，图像越相似
    ms_ssim_value = ms_ssim(image1, image2, data_range=1.0, size_average=True)  # data_range 表示像素值范围
    return ms_ssim_value.item()


Hnet = Ectformer(in_channel=24, out_channel=12)
Rnet = Ectformer(in_channel=12, out_channel=12)

Hnet.to(device)
Rnet.to(device)

params_trainable_H = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
params_trainable_R = (list(filter(lambda p: p.requires_grad, Rnet.parameters())))

optim = torch.optim.AdamW([{'params': Hnet.parameters()}, {'params': Rnet.parameters()}], lr=c.lr)

scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
                                             t_initial=c.epochs,
                                             lr_min=0,
                                             warmup_lr_init=c.warm_up_lr_init)

load(c.H_MODEL_PATH, Hnet)
load(c.R_MODEL_PATH, Rnet)

Hnet.eval()
Rnet.eval()

emd = ReversibleIHEMD(num_ensemble=10, max_imf=3)


with torch.no_grad():
    psnr_c_list = []
    psnr_s_list = []
    ssim_c_list = []
    ssim_s_list = []
    ms_ssim_c_list = []
    ms_ssim_s_list = []
    mae_c_list = []
    mae_s_list = []
    rmse_c_list = []
    rmse_s_list = []
    for i, data in tqdm(enumerate(datasets.testloader)):

        data = data.to(device)
        cover = data[data.shape[0] // 2:, :, :, :]
        secret = data[:data.shape[0] // 2, :, :, :]
        cover_input = emd.ihemd(cover)
        secret_input = emd.ihemd(secret)

        input = torch.cat([cover_input, secret_input], dim=1)

        #################
        #    forward:   #
        #################
        output = Hnet(input)

        steg_img = emd.reconstruct(output)

        #################
        #   backward:   #
        #################

        output_image = Rnet(output)

        secret_rev = emd.reconstruct(output_image)


        def deal_cpu(img):
            img = img.cpu().numpy().squeeze() * 255
            np.clip(img, 0, 255)
            return img


        psnr_c = computePSNR(deal_cpu(cover), deal_cpu(steg_img))
        psnr_s = computePSNR(deal_cpu(secret), deal_cpu(secret_rev))
        print("psnr_c:", psnr_c)
        print("psnr_s:", psnr_s)
        ssim_c = computeSSIM(deal_cpu(cover), deal_cpu(steg_img))
        ssim_s = computeSSIM(deal_cpu(secret), deal_cpu(secret_rev))
        mae_c = computemae(deal_cpu(cover), deal_cpu(steg_img))
        mae_s = computemae(deal_cpu(secret), deal_cpu(secret_rev))
        rmse_c = computermse(deal_cpu(cover), deal_cpu(steg_img))
        rmse_s = computermse(deal_cpu(secret), deal_cpu(secret_rev))

        ms_ssim_c = computermse(deal_cpu(cover), deal_cpu(steg_img))
        ms_ssim_s = computermse(deal_cpu(secret), deal_cpu(secret_rev))

        psnr_c_list.append(psnr_c)
        psnr_s_list.append(psnr_s)
        ssim_c_list.append(ssim_c)
        ssim_s_list.append(ssim_s)
        ms_ssim_c_list.append(ssim_c)
        ms_ssim_s_list.append(ssim_s)
        mae_c_list.append(mae_c)
        mae_s_list.append(mae_s)
        rmse_c_list.append(rmse_c)
        rmse_s_list.append(rmse_s)
        if i > 1000:
            break
        # print(psnr_c,psnr_s)

        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_stego + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)

    print("PSNR:", np.mean(psnr_c_list), np.mean(psnr_s_list))
    print("SSIM:", np.mean(ssim_c_list), np.mean(ssim_s_list))
    print("MS_SSIM:", np.mean(ms_ssim_c_list), np.mean(ms_ssim_s_list))
    print("MAE:", np.mean(mae_c_list), np.mean(mae_s_list))
    print("RMSE:", np.mean(rmse_c_list), np.mean(rmse_s_list))

# print(torch.__version__)