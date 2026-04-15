import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import config as c
import datasets
import modules.Unet_common as common
import timm.scheduler
import cv2
from models.ihemd import ReversibleIHEMD
from models.ectformer import Ectformer
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio
from tqdm import tqdm
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

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
    # print(img1.shape)
    # print(img2.shape)

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


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2)
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


Hnet = Ectformer(in_channel=48, out_channel=12)
Rnet = Ectformer(in_channel=12, out_channel=36)

Hnet.to(device)
Rnet.to(device)

params_trainable_H = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
params_trainable_R = (list(filter(lambda p: p.requires_grad, Rnet.parameters())))

optim = torch.optim.AdamW([{'params': Hnet.parameters()}, {'params': Rnet.parameters()}], lr=c.lr)

scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
                                             t_initial=c.epochs,
                                             lr_min=0,
                                             warmup_t=c.warm_up_epoch,
                                             warmup_lr_init=c.warm_up_lr_init)

load(c.H_MODEL_PATH, Hnet)
load(c.R_MODEL_PATH, Rnet)

Hnet.eval()
Rnet.eval()

emd = ReversibleIHEMD(num_ensemble=10, max_imf=3)

dwt = common.DWT()
iwt = common.IWT()

with torch.no_grad():
    psnr_c_list = []
    ssim_c_list = []
    psnr_1s_list = []
    ssim_1s_list = []
    psnr_2s_list = []
    ssim_2s_list = []
    psnr_3s_list = []
    ssim_3s_list = []

    i = 0
    for img in tqdm(datasets.DIV2K_multi_val_loader, total=len(datasets.DIV2K_multi_val_loader)):
        img = img.to(device)
        bs = c.multi_batch_szie_test
        cover = img[0:bs, :, :, :]
        secret_o = img[bs:, :, :, :]
        secret = rearrange(secret_o, '(b n) c h w -> b (n c) h w', n=c.num_secret)


        secret1 = secret[:, :3, :, :]
        secret2 = secret[:, 3:6, :, :]
        secret3 = secret[:, 6:9, :, :]

        cover_input = emd.ihemd(cover)
        secret_input1 = emd.ihemd(secret1)
        secret_input2 = emd.ihemd(secret2)
        secret_input3 = emd.ihemd(secret3)
        secret_input4 = torch.cat([secret_input1, secret_input2], dim=1)
        secret_input = torch.cat([secret_input4, secret_input3], dim=1)

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
        # print(output_image.shape)
        output_image1 = output_image[:, :12, :, :]
        output_image2 = output_image[:, 12:24, :, :]
        output_image3 = output_image[:, 24:36, :, :]
        # print(output_image3.shape)

        secret_rev1 = emd.reconstruct(output_image1)
        secret_rev2 = emd.reconstruct(output_image2)
        secret_rev3 = emd.reconstruct(output_image3)

        secret_rev4 = torch.cat((secret_rev1, secret_rev2), dim=1)
        secret_rev = torch.cat((secret_rev4, secret_rev3), dim=1)


        # secret_rev = output_image

        def deal_cpu(img):
            img = img.cpu().numpy().squeeze() * 255
            np.clip(img, 0, 255)
            return img


        psnr_c = computePSNR(deal_cpu(cover), deal_cpu(steg_img))
        ssim_c = computeSSIM(deal_cpu(cover), deal_cpu(steg_img))

        psnr_1s = computePSNR(deal_cpu(secret1), deal_cpu(secret_rev1))
        ssim_1s = computeSSIM(deal_cpu(secret1), deal_cpu(secret_rev1))

        psnr_2s = computePSNR(deal_cpu(secret2), deal_cpu(secret_rev2))
        ssim_2s = computeSSIM(deal_cpu(secret2), deal_cpu(secret_rev2))

        psnr_3s = computePSNR(deal_cpu(secret3), deal_cpu(secret_rev3))
        ssim_3s = computeSSIM(deal_cpu(secret3), deal_cpu(secret_rev3))

        psnr_c_list.append(psnr_c)
        ssim_c_list.append(ssim_c)

        psnr_1s_list.append(psnr_1s)
        ssim_1s_list.append(ssim_1s)

        psnr_2s_list.append(psnr_2s)
        ssim_2s_list.append(ssim_2s)

        psnr_3s_list.append(psnr_3s)
        ssim_3s_list.append(ssim_3s)

        if i > 1000:
            break
        # print(psnr_c,psnr_s)

        torchvision.utils.save_image(cover, c.IMAGE3_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret1, c.IMAGE3_PATH_secret1 + '%.5d.png' % i)
        torchvision.utils.save_image(secret2, c.IMAGE3_PATH_secret2 + '%.5d.png' % i)
        torchvision.utils.save_image(secret3, c.IMAGE3_PATH_secret3 + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE3_PATH_stego + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev1, c.IMAGE3_PATH_secret_rev1 + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev2, c.IMAGE3_PATH_secret_rev2 + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev3, c.IMAGE3_PATH_secret_rev3 + '%.5d.png' % i)
        i = i + 1

    print("PSNR:", np.mean(psnr_c_list), np.mean(psnr_1s_list), np.mean(psnr_2s_list), np.mean(psnr_3s_list))
    print("SSIM:", np.mean(ssim_c_list), np.mean(ssim_1s_list), np.mean(ssim_2s_list), np.mean(ssim_3s_list))


