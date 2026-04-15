import torch
import torch.nn
import torch.optim
import math
import numpy as np
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import logging
import util

from models.ectformer import Ectformer

import timm.scheduler
from models.ihemd import ReversibleIHEMD

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm


def Hiding_loss(X, Y):
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + 1e-6)
    loss = torch.mean(error)
    return loss.to(device)


def Revealing_loss(X, Y):
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + 1e-6)
    loss = torch.mean(error)
    return loss.to(device)


def Intrinsic_Loss(X, Y):
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + 1e-6)
    loss = torch.mean(error)
    return loss.to(device)

# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


param = {
        'nimfs': 3,
        'tol': 0.05,
        'type': 6
    }
emd = ReversibleIHEMD(num_ensemble=10, max_imf=3)



Hnet = Ectformer(in_channel=24, out_channel=12)
Rnet = Ectformer(in_channel=12, out_channel=12)

Hnet.to(device)
Hnet.apply(weights_init)

Rnet.to(device)
Rnet.apply(weights_init)

params_trainable_H = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
params_trainable_R = (list(filter(lambda p: p.requires_grad, Rnet.parameters())))

optim = torch.optim.AdamW([{'params': Hnet.parameters()}, {'params': Rnet.parameters()}], lr=c.lr)

scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
                                             t_initial=c.epochs,
                                             lr_min=0,
                                             warmup_t=c.warm_up_epoch,
                                             warmup_lr_init=c.warm_up_lr_init)




if c.tain_next:
    load(c.H_MODEL_PATH, Hnet, optim)
    load(c.R_MODEL_PATH, Rnet, optim)


util.setup_logger('train', './', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')
logger_train.info(Hnet)
logger_train.info(Rnet)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")
    for i_epoch in range(c.epochs):
        scheduler.step(i_epoch + c.train_next)
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        c_loss_history = []
        r_loss_history = []
        i_loss_history = []
        #################
        #     train:    #
        #################

        for i_batch, (cover, secret) in tqdm(enumerate(zip(datasets.train1loader, datasets.train2loader)),total=len(datasets.train1loader)):
            cover = cover.to(device)
            secret = secret.to(device)

            cover_input = emd.ihemd(cover)
            secret_input = emd.ihemd(secret)


            input = torch.cat([cover, secret], dim=1)

            #################
            #    forward:   #
            #################
            output = Hnet(input)
            stego_img = emd.reconstruct(output)

            #################
            #   backward:   #
            #################

            output_image = Rnet(output)

            secret_rev = emd.reconstruct(output_image)

            #################
            #     loss:     #
            #################
            c_loss = Hiding_loss(cover.cuda(), stego_img.cuda())
            r_loss = Revealing_loss(secret_rev.cuda(), secret.cuda())
            i_loss = Intrinsic_Loss(cover_input.cuda(), output.cuda())+Intrinsic_Loss(secret_input.cuda(), output_image.cuda())
            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * i_loss

            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])

            c_loss_history.append([c_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            i_loss_history.append([i_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
        c_epoch_losses = np.mean(np.array(c_loss_history), axis=0)
        i_epoch_losses = np.mean(np.array(i_loss_history), axis=0)

        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                Hnet.eval()
                Rnet.eval()
                for x in tqdm(datasets.testloader,total=len(datasets.testloader)):
                    x = x.to(device)
                    cover = x[x.shape[0] // 2:, :, :, :]
                    secret = x[:x.shape[0] // 2, :, :, :]

                    cover_input = emd.ihemd(cover)
                    secret_input = emd.ihemd(secret)

                    input = torch.cat([cover, secret], dim=1)

                    #################
                    #    forward:   #
                    #################

                    output = Hnet(input)

                    stego = emd.reconstruct(output)


                    #################
                    #   backward:   #
                    #################

                    output_image = Rnet(output)

                    secret_rev = emd.reconstruct(output_image)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, stego)
                    psnr_c.append(psnr_temp_c)

                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                )
        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
        logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
            f'r_Loss: {r_epoch_losses[0].item():.4f} | '
            f'c_Loss: {c_epoch_losses[0].item():.4f} | '
            f'i_Loss: {i_epoch_losses[0].item():.4f} | '
        )

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': Hnet.state_dict()}, c.MODEL_PATH + 'Hnet_model_checkpoint_%.5i' % i_epoch + '.pt')
            torch.save({'opt': optim.state_dict(),
                        'net': Rnet.state_dict()}, c.MODEL_PATH + 'Rnet_model_checkpoint_%.5i' % i_epoch + '.pt')

    torch.save({'opt': optim.state_dict(),
                'net': Hnet.state_dict()}, c.MODEL_PATH + 'Hnet_model' + '.pt')
    torch.save({'opt': optim.state_dict(),
                'net': Rnet.state_dict()}, c.MODEL_PATH + 'Rnet_model' + '.pt')

    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': Hnet.state_dict()}, c.MODEL_PATH + 'Hnet_model_abort' + '.pt')
        torch.save({'opt': optim.state_dict(),
                    'net': Rnet.state_dict()}, c.MODEL_PATH + 'Rnet_model_abort' + '.pt')
    raise

finally:
    viz.signal_stop()