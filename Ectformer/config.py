
# 10 ** log10_lr
epochs = 8000
weight_decay = 1e-5
init_scale = 0.01

# optimer config
lr = 0.0001
warm_up_epoch = 20
warm_up_lr_init = 5e-6
train_next = 0

lamda_reconstruction =1
lamda_guide = 1
lamda_restrict_loss = 1
device_ids = [0]

# Train:
batch_size = 12
cropsize = 256
betas = (0.5, 0.999)
weight_step = 500
gamma = 0.5


num_secret = 3
multi_batch_szie = 6
multi_batch_iteration = (num_secret + 1) * 6
test_multi_batch_size = num_secret + 1
multi_batch_szie_test = 1
multi_batch_iteration_test = (num_secret + 1) * 1


# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 1


# Dataset
TRAIN_PATH = r'D:\data\div\train'
VAL_PATH = r'D:\data\div\val'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = True


# Saving checkpoints:

MODEL_PATH = './model/'
checkpoint_on_error = True
SAVE_freq = 100

IMAGE_PATH = 'D:/Ectformer/image/image1/div2k/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_stego = IMAGE_PATH + 'stego/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

IMAGE2_PATH = 'D:/Ectformer/image/image2/'
IMAGE2_PATH_cover = IMAGE2_PATH + 'cover/'
IMAGE2_PATH_secret1 = IMAGE2_PATH + 'secret1/'
IMAGE2_PATH_secret2 = IMAGE2_PATH + 'secret2/'
IMAGE2_PATH_stego = IMAGE2_PATH + 'stego/'
IMAGE2_PATH_secret_rev1 = IMAGE2_PATH + 'secret-rev1/'
IMAGE2_PATH_secret_rev2 = IMAGE2_PATH + 'secret-rev2/'

IMAGE3_PATH = 'D:/Ectformer/image/image3/'
IMAGE3_PATH_cover = IMAGE3_PATH + 'cover/'
IMAGE3_PATH_secret1 = IMAGE3_PATH + 'secret1/'
IMAGE3_PATH_secret2 = IMAGE3_PATH + 'secret2/'
IMAGE3_PATH_secret3 = IMAGE3_PATH + 'secret3/'
IMAGE3_PATH_stego = IMAGE3_PATH + 'stego/'
IMAGE3_PATH_secret_rev1 = IMAGE3_PATH + 'secret-rev1/'
IMAGE3_PATH_secret_rev2 = IMAGE3_PATH + 'secret-rev2/'
IMAGE3_PATH_secret_rev3 = IMAGE3_PATH + 'secret-rev3/'

IMAGE4_PATH = 'D:/Ectformer/image/image4/'
IMAGE4_PATH_cover = IMAGE4_PATH + 'cover/'
IMAGE4_PATH_secret1 = IMAGE4_PATH + 'secret1/'
IMAGE4_PATH_secret2 = IMAGE4_PATH + 'secret2/'
IMAGE4_PATH_secret3 = IMAGE4_PATH + 'secret3/'
IMAGE4_PATH_secret4 = IMAGE4_PATH + 'secret4/'
IMAGE4_PATH_stego = IMAGE4_PATH + 'stego/'
IMAGE4_PATH_secret_rev1 = IMAGE4_PATH + 'secret-rev1/'
IMAGE4_PATH_secret_rev2 = IMAGE4_PATH + 'secret-rev2/'
IMAGE4_PATH_secret_rev3 = IMAGE4_PATH + 'secret-rev3/'
IMAGE4_PATH_secret_rev4 = IMAGE4_PATH + 'secret-rev4/'



# Load:
suffix = 'model.pt'
H_MODEL_PATH=r'D:\Ectformer\result\div_xin\Hnet_model_checkpoint_08000.pt'
R_MODEL_PATH=r'D:\Ectformer\result\div_xin\Rnet_model_checkpoint_08000.pt'
tain_next = False
trained_epoch = 1


norm_train = 'c'