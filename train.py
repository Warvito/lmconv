import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from IPython import embed
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

from layers import PONO
from masking import get_generation_order_idx, get_masks
from model import OurPixelCNN
from utils import configure_logger, discretized_mix_logistic_loss_1d, discretized_mix_logistic_loss_1d_averaged, \
    sample_from_discretized_mix_logistic_1d, EMA

lr = 0.0002
batch_size = 8
clip = 2e6
kernel_size = 3
max_dilation = 2
nr_logistic_mix = 10
n_bits = 8
data_dir = "./data"
nr_resnet = 5
nr_filters = 160
dropout_prob = 0.0
weight_decay = 0
lr_decay = 0.999995
ema = 1
max_epochs = 20
test_interval = 1
save_interval = 1
exp_id = 1
sample_interval = 1
print_every = 20
order = "s_curve"
sample_batch_size = 8
dataset_obs = (1, 28, 28)

# Set seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

run_dir = os.path.join("runs", "mnist")
os.makedirs(run_dir, exist_ok=True)

# Log arguments
timestamp = time.strftime("%Y%m%d-%H%M%S")
logfile = f"train_{timestamp}.log"
logger = configure_logger(os.path.join(run_dir, logfile))
logger.info("Run directory: %s", run_dir)

# Create data loaders
input_channels = dataset_obs[0]
data_loader_kwargs = {'num_workers': 6, 'pin_memory': True, 'drop_last': True, 'batch_size': batch_size}

rescaling = lambda x: (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
rescaling_inv = lambda x: .5 * x + .5
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

train_ds = datasets.MNIST(data_dir, download=True, train=True, transform=ds_transforms)
test_ds = datasets.MNIST(data_dir, train=False, transform=ds_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **data_loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_ds, **data_loader_kwargs)

# Losses for 1-channel images
loss_op = discretized_mix_logistic_loss_1d
loss_op_averaged = discretized_mix_logistic_loss_1d_averaged
sample_op = lambda x, i, j: sample_from_discretized_mix_logistic_1d(x, i, j, nr_logistic_mix)

# Construct model
logger.info("Constructing our model")
norm_op = lambda num_channels: PONO()
model = OurPixelCNN(
    nr_resnet=nr_resnet,
    nr_filters=nr_filters,
    input_channels=input_channels,
    nr_logistic_mix=nr_logistic_mix,
    kernel_size=(kernel_size, kernel_size),
    max_dilation=max_dilation,
    feature_norm_op=norm_op,
    dropout_prob=dropout_prob
)

# Get generation orders
generation_idx = get_generation_order_idx(order, dataset_obs[1], dataset_obs[2])

# Make masks
masks = get_masks(generation_idx, dataset_obs[1], dataset_obs[2], kernel_size, max_dilation, observed_idx=None)

model = model.cuda()

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

# Initialize exponential moving average of parameters
if ema < 1:
    ema = EMA(ema)
    ema.register(model)

def test(model, all_masks, test_loader, epoch="N/A"):
    test_loss = 0.
    pbar = tqdm.tqdm(test_loader, desc=f"Test after epoch {epoch}")
    for batch_idx, (input, _) in enumerate(pbar):
        input = input.cuda()

        # Average likelihoods over multiple orderings
        outputs = []
        for mask_init, mask_undilated, mask_dilated in all_masks:
            output = model(input, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            outputs.append(output)

        loss = loss_op_averaged(input, outputs)

        test_loss += loss.item()
        del loss, output

        pbar.set_description(f"Test after epoch {epoch} {test_loss}")

    return test_loss


def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, obs):
    model.eval()
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()

    for n_pix, (i, j) in enumerate(tqdm.tqdm(generation_idx, desc="Sampling pixels")):
        out = model(data, sample=True, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        out_sample = sample_op(out, i, j)
        data[:, :, i, j] = out_sample
        if (n_pix <= 256 and n_pix % 32 == 0) or n_pix % 256 == 0:
            sample_save_path = os.path.join(run_dir, f'train_full_order_s_oredr_{n_pix}of{len(generation_idx)}pix.png')
            utils.save_image(rescaling_inv(data), sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)

    data = rescaling_inv(data).cpu()
    return data


writer = SummaryWriter(log_dir=run_dir)
global_step = 0
min_train_bpd = 1e12
min_test_bpd_by_obs = 1e12
last_saved_epoch = -1
for epoch in range(1, max_epochs):
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input, _) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}")):
        input = input.cuda()  # [-1, 1] range images

        mask_init, mask_undilated, mask_dilated = masks
        loss = model.loss(input, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)

        optimizer.zero_grad()
        loss.backward()
        # Compute and rescale gradient norm
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
        writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
        optimizer.step()
        if ema < 1:
            ema.update(model)
        train_loss += loss.item()

        if (batch_idx + 1) % print_every == 0:
            logger.info('train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
                train_loss / print_every,
                (time.time() - time_),
                global_step))
            train_loss = 0.
            time_ = time.time()

    # decrease learning rate
    scheduler.step()

    model.eval()
    with torch.no_grad():
        save_dict = {}

        if (epoch + 1) % test_interval == 0:
            # test with all masks
            test_loss = test(model,
                            [masks],
                            test_loader,
                            epoch)

        # Save checkpoint so we have checkpoints every save_interval epochs, as well as a rolling most recent checkpoint
        save_path = os.path.join(run_dir, f"ep{epoch}.pth")
        logger.info('saving model to %s...', save_path)
        save_dict["epoch"] = epoch
        save_dict["global_step"] = global_step
        try:
            save_dict["model_state_dict"] = model.state_dict()
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
            if ema < 1:
                save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, save_path)
        except Exception as e:
            logger.error("Failed to save checkpoint! Error: %s", e)

        if (epoch + 1) % sample_interval == 0:
            try:
                all_masks = [masks]
                sample_order_i = np.random.randint(len(all_masks))

                logger.info('sampling images with observation ordering variant %d...', sample_order_i)
                sample_t = sample(model,
                                  generation_idx,
                                  *all_masks[sample_order_i],
                                  input.shape[1:])
                sample_save_path = os.path.join(run_dir, f"tsample_obs_{epoch}_order{sample_order_i}.png")
                utils.save_image(sample_t, sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)
            except Exception as e:
                logger.error("Failed to sample images! Error: %s", e)
