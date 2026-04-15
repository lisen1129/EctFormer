import torch
import torch.nn
import torch.optim
import math
import numpy as np
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import warnings
import logging
import util

from models.ectformer import Ectformer
import timm.scheduler
from models.ihemd import ReversibleIHEMD

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from einops import rearrange



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

Hnet = Ectformer(in_channel=60, out_channel=12)
Rnet = Ectformer(in_channel=12, out_channel=48)

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

        for i_batch, img in tqdm(enumerate(datasets.DIV2K_multi_train_loader),total=len(datasets.DIV2K_multi_train_loader)):
            img = img.to(device)
            bs = c.multi_batch_szie
            cover = img[0:bs, :, :, :]
            secret_o = img[bs:, :, :, :]
            secret = rearrange(secret_o, '(b n) c h w -> b (n c) h w', n=c.num_secret)

            secret1 = secret[:, :3, :, :]
            secret2 = secret[:, 3:6, :, :]
            secret3 = secret[:, 6:9, :, :]
            secret4 = secret[:, 9:12, :, :]

            cover_input = emd.ihemd(cover)
            secret_input1 = emd.ihemd(secret1)
            secret_input2 = emd.ihemd(secret2)
            secret_input3 = emd.ihemd(secret3)
            secret_input4 = emd.ihemd(secret4)
            secret_input5 = torch.cat([secret_input1, secret_input2], dim=1)
            secret_input6 = torch.cat([secret_input5, secret_input3], dim=1)
            secret_input = torch.cat([secret_input6, secret_input4], dim=1)
            input = torch.cat([cover_input, secret_input], dim=1)
            #################
            #    forward:   #
            #################
            output = Hnet(input)

            stego_img = emd.reconstruct(output)


            #################
            #   backward:   #
            #################
            output_image = Rnet(output)
            # print(output_image.shape)
            output_image1 = output_image[:, :12, :, :]
            output_image2 = output_image[:, 12:24, :, :]
            output_image3 = output_image[:, 24:36, :, :]
            output_image4 = output_image[:, 36:48, :, :]
            # print(output_image3.shape)

            secret_rev1 = emd.reconstruct(output_image1)
            secret_rev2 = emd.reconstruct(output_image2)
            secret_rev3 = emd.reconstruct(output_image3)
            secret_rev4 = emd.reconstruct(output_image4)

            secret_rev5 = torch.cat((secret_rev1, secret_rev2), dim=1)
            secret_rev6 = torch.cat((secret_rev5, secret_rev3), dim=1)
            secret_rev = torch.cat((secret_rev6, secret_rev4), dim=1)

            #################
            #     loss:     #
            #################
            c_loss = Hiding_loss(cover.cuda(), steg_img.cuda())

            i_loss = Intrinsic_Loss(secret_rev1.cuda(), secret1.cuda())+Intrinsic_Loss(secret_rev2.cuda(), secret2.cuda()) + Intrinsic_Loss(output_image.cuda(), secret_input.cuda())+Intrinsic_Loss(secret_rev3.cuda(), secret3.cuda())+Intrinsic_Loss(secret_rev4.cuda(), secret4.cuda())

            r_loss = Revealing_loss(secret_rev.cuda(), secret.cuda())

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * c_loss + c.lamda_restrict_loss * i_loss


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
                psnr_c = []
                psnr_1s = []
                psnr_2s = []
                psnr_3s = []
                psnr_4s = []
                Hnet.eval()
                Rnet.eval()
                for img in tqdm(datasets.DIV2K_multi_val_loader, total=len(datasets.DIV2K_multi_val_loader)):
                    img = img.to(device)
                    bs = c.multi_batch_szie_test
                    cover = img[0:bs, :, :, :]
                    secret_o = img[bs:, :, :, :]
                    secret = rearrange(secret_o, '(b n) c h w -> b (n c) h w', n=c.num_secret)

                    secret1 = secret[:, :3, :, :]
                    secret2 = secret[:, 3:6, :, :]
                    secret3 = secret[:, 6:9, :, :]
                    secret4 = secret[:, 9:12, :, :]

                    cover_input = emd.ihemd(cover)
                    secret_input1 = emd.ihemd(secret1)
                    secret_input2 = emd.ihemd(secret2)
                    secret_input3 = emd.ihemd(secret3)
                    secret_input4 = emd.ihemd(secret4)
                    secret_input5 = torch.cat([secret_input1, secret_input2], dim=1)
                    secret_input6 = torch.cat([secret_input5, secret_input3], dim=1)
                    secret_input = torch.cat([secret_input6, secret_input4], dim=1)

                    input = torch.cat([cover_input, secret_input], dim=1)

                    #################
                    #    forward:   #
                    #################

                    output = Hnet(input)

                    steg_img = emd.reconstruct(output)

                    if c.norm_train == 'clamp':
                        encode_img_c = torch.clamp(output, 0, 1)
                    else:
                        encode_img_c = output

                    #################
                    #   backward:   #
                    #################
                    output_image = Rnet(encode_img_c)
                    # print(output_image.shape)
                    output_image1 = output_image[:, :12, :, :]
                    output_image2 = output_image[:, 12:24, :, :]
                    output_image3 = output_image[:, 24:36, :, :]
                    output_image4 = output_image[:, 36:48, :, :]
                    # print(output_image3.shape)

                    secret_rev1 = emd.reconstruct(output_image1)
                    secret_rev2 = emd.reconstruct(output_image2)
                    secret_rev3 = emd.reconstruct(output_image3)
                    secret_rev4 = emd.reconstruct(output_image4)

                    secret_rev5 = torch.cat((secret_rev1, secret_rev2), dim=1)
                    secret_rev6 = torch.cat((secret_rev5, secret_rev3), dim=1)
                    secret_rev = torch.cat((secret_rev6, secret_rev4), dim=1)


                    def deal_cpu(img):
                        img = img.cpu().numpy().squeeze() * 255
                        np.clip(img, 0, 255)
                        return img


                    psnr_temp_c = computePSNR(deal_cpu(cover), deal_cpu(steg_img))
                    psnr_temp_1s = computePSNR(deal_cpu(secret1), deal_cpu(secret_rev1))
                    psnr_temp_2s = computePSNR(deal_cpu(secret2), deal_cpu(secret_rev2))
                    psnr_temp_3s = computePSNR(deal_cpu(secret3), deal_cpu(secret_rev3))
                    psnr_temp_4s = computePSNR(deal_cpu(secret4), deal_cpu(secret_rev4))


                    psnr_c.append(psnr_temp_c)
                    psnr_1s.append(psnr_temp_1s)
                    psnr_2s.append(psnr_temp_2s)
                    psnr_3s.append(psnr_temp_3s)
                    psnr_4s.append(psnr_temp_4s)



                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                writer.add_scalars("PSNR_1S", {"average psnr": np.mean(psnr_1s)}, i_epoch)
                writer.add_scalars("PSNR_2S", {"average psnr": np.mean(psnr_2s)}, i_epoch)
                writer.add_scalars("PSNR_3S", {"average psnr": np.mean(psnr_3s)}, i_epoch)
                writer.add_scalars("PSNR_4S", {"average psnr": np.mean(psnr_4s)}, i_epoch)

                logger_train.info(
                    f"TEST:   "
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                    f'PSNR_1S: {np.mean(psnr_1s):.4f} | '
                    f'PSNR_2S: {np.mean(psnr_2s):.4f} | '
                    f'PSNR_3S: {np.mean(psnr_3s):.4f} | '
                    f'PSNR_4S: {np.mean(psnr_4s):.4f} | '
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