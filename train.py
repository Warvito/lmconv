from IPython import embed
import argparse
import itertools
from operator import itemgetter
import os
import re
import time

from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
import tqdm

from layers import PONO
from masking import *
from model import OurPixelCNN
from utils import *

lr=0.0002
batch_size=32
clip=2e6
kernel_size=3
max_dilation=2
nr_logistic_mix = 10
n_bits = 8
data_dir="./data"
nr_resnet=5
nr_filters=160
dropout_prob=0.0
weight_decay=0
lr_decay=0.999995
ema=1
max_epochs=20
test_interval=1
save_interval=1
exp_id=1
sample_interval=1

# Set seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

dataset_name = "mnist"
_name = "mnist_lr{:.5f}_bs{}_gc{}_k{}_md{}".format(lr, batch_size, clip, kernel_size, max_dilation)
run_dir = os.path.join("runs", _name)
os.makedirs(run_dir, exist_ok=True)

# Log arguments
timestamp = time.strftime("%Y%m%d-%H%M%S")
logfile = f"train_{timestamp}.log"
logger = configure_logger(os.path.join(run_dir, logfile))
logger.info("Run directory: %s", run_dir)

# Create data loaders
sample_batch_size = 16
dataset_obs = (1, 28, 28)
input_channels = dataset_obs[0]
data_loader_kwargs = {'num_workers':6, 'pin_memory':True, 'drop_last':True, 'batch_size':batch_size}

# Create data loaders
rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
rescaling_inv = lambda x : .5 * x + .5
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, download=True, train=True, transform=ds_transforms), shuffle=True, **data_loader_kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False, transform=ds_transforms), **data_loader_kwargs)

# Default upper bounds for progress bars
train_total = None
test_total = None

# Losses for 1-channel images
loss_op = discretized_mix_logistic_loss_1d
loss_op_averaged = discretized_mix_logistic_loss_1d_averaged
sample_op = lambda x, i, j: sample_from_discretized_mix_logistic_1d(x, i, j,nr_logistic_mix)


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
            dropout_prob=dropout_prob)

all_generation_idx_by_obs = {}
all_masks_by_obs = {}

# Get generation orders
all_generation_idx = []
order="s_curve"
base_generation_idx = get_generation_order_idx(order, dataset_obs[1], dataset_obs[2])
# if args.base_order_transpose:
#     base_generation_idx = transpose(base_generation_idx)
# if args.base_order_reflect_rows:
#     base_generation_idx = reflect_rows(base_generation_idx, dataset_obs)
# if args.base_order_reflect_cols:
#     base_generation_idx = reflect_cols(base_generation_idx, dataset_obs)
all_generation_idx.append(base_generation_idx)

# Generate center square last for inpainting
observed_idx = None

# Plot orders
plot_orders_out_path = os.path.join(run_dir, f"train_orderings.png")

try:
    plot_orders(all_generation_idx, dataset_obs, size=5, plot_rows=min(len(all_generation_idx), 4), out_path=plot_orders_out_path)
except IndexError as e:
    logger.error("Failed to plot orders: %s", e)

all_generation_idx_by_obs[dataset_obs] = all_generation_idx

# Make masks and plot
all_masks = []
for i, generation_idx in enumerate(all_generation_idx):
    masks = get_masks(generation_idx, dataset_obs[1], dataset_obs[2], kernel_size, max_dilation,
                      observed_idx=observed_idx,
                      out_dir=run_dir,
                      plot_suffix=f"obs_order{i}")
    logger.info(f"Mask shapes: {masks[0].shape}, {masks[1].shape}, {masks[2].shape}")
    all_masks.append(masks)
all_masks_by_obs[dataset_obs] = all_masks

model = model.cuda()

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

model = nn.DataParallel(model)
checkpoint_epochs = -1
checkpoint_step = -1

# Initialize exponential moving average of parameters
if ema < 1:
    ema = EMA(ema)
    ema.register(model.module)


def test(model, all_masks, test_loader, epoch="N/A", progress_bar=True,
         slice_op=None, sliced_obs=dataset_obs):
    logger.info(f"Testing with ensemble of {len(all_masks)} orderings")
    test_loss = 0.
    pbar = tqdm.tqdm(test_loader,
                     desc=f"Test after epoch {epoch}",
                     disable=(not progress_bar),
                     total=test_total)
    num_images = 0
    for batch_idx, (input,_) in enumerate(pbar):
        num_images += input.shape[0]

        input = input.cuda(non_blocking=True)
        input_var = Variable(input)

        # Average likelihoods over multiple orderings
        outputs = []
        for mask_init, mask_undilated, mask_dilated in all_masks:
            output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            output = slice_op(output) if slice_op is not None else output
            outputs.append(output)

        order_prefix = "_".join(args.order)
        np.save(f"{dataset_name}_{order_prefix}_all_generation_idx", all_generation_idx)

        input_var_for_loss = slice_op(input_var) if slice_op is not None else input_var
        loss = loss_op_averaged(input_var_for_loss, outputs)

        test_loss += loss.item()
        del loss, output

        deno = num_images * np.prod(sliced_obs) * np.log(2.)
        pbar.set_description(f"Test after epoch {epoch} {test_loss / deno}")

    deno = num_images * np.prod(sliced_obs) * np.log(2.)
    assert deno > 0, embed()
    test_bpd = test_loss / deno
    return test_bpd


def get_sampling_images(loader):
    # Get batch of images to complete for inpainting, or None for --sample_region=full
    if args.sample_region == "full":
        return None

    logger.info('getting batch of images to complete...')
    # Get sample_batch_size images from test set
    batches_to_complete = []
    sample_iter = iter(loader)
    for _ in range(sample_batch_size // args.batch_size + 1):
        batches_to_complete.append(next(sample_iter)[0])  # ignore labels
    del sample_iter

    batch_to_complete = torch.cat(batches_to_complete, dim=0)[:sample_batch_size]
    logger.info('got %d images to complete with shape %s', len(batch_to_complete), batch_to_complete.shape)

    return batch_to_complete


def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, batch_to_complete, obs):
    model.eval()
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    sample_idx = generation_idx
    context = None
    batch_to_complete = None

    logger.info(f"Before sampling, data has range {data.min().item()}-{data.max().item()} (mean {data.mean().item()}), dtype={data.dtype} {type(data)}")
    for n_pix, (i, j) in enumerate(tqdm.tqdm(sample_idx, desc="Sampling pixels")):
        data_v = Variable(data)
        t1 = time.time()
        out = model(data_v, sample=True, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        t2 = time.time()
        out_sample = sample_op(out, i, j)
        logger.info("%d %d,%d Time to infer logits=%f s, sample=%f s", n_pix, i, j, t2-t1, time.time()-t2)
        data[:, :, i, j] = out_sample
        logger.info(f"Sampled pixel {i},{j}, with batchwise range {out_sample.min().item()}-{out_sample.max().item()} (mean {out_sample.mean().item()}), dtype={out_sample.dtype} {type(out_sample)}")

        if (n_pix <= 256 and n_pix % 32 == 0) or n_pix % 256 == 0:
            sample_save_path = os.path.join(run_dir, f'{args.mode}_{args.sample_region}_{args.sample_size_h}x{args.sample_size_w}_o1{args.sample_offset1}_o2{args.sample_offset2}_obs{obs2str(obs)}_ep{checkpoint_epochs}_order{sample_order_i}_{n_pix}of{len(sample_idx)}pix.png')
            utils.save_image(rescaling_inv(data), sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)

    data = rescaling_inv(data).cpu()

    if batch_to_complete is not None and context is not None:
        # Interleave along batch dimension to visualize GT images
        difference = torch.abs(data - batch_to_complete)
        logger.info(f"Context range {context.min()}-{context.max()}. Data range {data.min()}-{data.max()}. batch_to_complete range {batch_to_complete.min()}-{batch_to_complete.max()}")
        data = torch.stack([context, data, batch_to_complete, difference], dim=1).view(-1, *data.shape[1:])

    return data


logger.info("starting training")
writer = SummaryWriter(log_dir=run_dir)
global_step = checkpoint_step + 1
min_train_bpd = 1e12
min_test_bpd_by_obs = 1e12
last_saved_epoch = -1
for epoch in range(1, max_epochs):
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}", total=train_total)):
        input = input.cuda(non_blocking=True)  # [-1, 1] range images

        obs = input.shape[1:]
        all_masks = all_masks_by_obs[obs]
        order_i = np.random.randint(len(all_masks))
        mask_init, mask_undilated, mask_dilated = all_masks[order_i]
        output = model(input, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)

        loss = loss_op(input, output)
        deno = batch_size * np.prod(obs) * np.log(2.)
        assert deno > 0, embed()
        train_bpd = loss / deno

        optimizer.zero_grad()
        loss.backward()
        # Compute and rescale gradient norm
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
        writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
        optimizer.step()
        if ema < 1:
            ema.update(model.module)
        train_loss += loss.item()

        writer.add_scalar('train/bpd', train_bpd.item(), global_step)
        min_train_bpd = min(min_train_bpd, train_bpd.item())
        writer.add_scalar('train/min_bpd', min_train_bpd, global_step)

        if batch_idx >= 100 and train_bpd.item() >= 10:
            logger.warning("WARNING: main.py: large batch loss {} bpd".format(train_bpd.item()))

        if (batch_idx + 1) % args.print_every == 0:
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            average_bpd = train_loss / args.print_every if args.minimize_bpd else train_loss / deno
            logger.info('train bpd : {:.4f}, train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
                average_bpd,
                train_loss / args.print_every,
                (time.time() - time_),
                global_step))
            train_loss = 0.
            time_ = time.time()

    # decrease learning rate
    scheduler.step()

    model.eval()
    with torch.no_grad():
        save_dict = {}

        if (epoch + 1) % args.test_interval == 0:
                # test with all masks
                test_bpd = test(model,
                                all_masks_by_obs[obs],
                                test_loader,
                                epoch,
                                progress_bar=True)
                writer.add_scalar(f'test/bpd', test_bpd, global_step)
                logger.info(f"test loss : %s bpd" % test_bpd)
                save_dict[f"test_loss"] = test_bpd

                # Log min test bpd for smoothness
                min_test_bpd_by_obs[obs] = min(min_test_bpd_by_obs[obs], test_bpd)
                writer.add_scalar(f'test/min_bpd', min_test_bpd_by_obs[obs], global_step)
                if obs == dataset_obs:
                    writer.add_scalar(f'test/bpd', test_bpd, global_step)
                    writer.add_scalar(f'test/min_bpd', min_test_bpd_by_obs[obs], global_step)

        # Save checkpoint so we have checkpoints every save_interval epochs, as well as a rolling most recent checkpoint
        save_path = os.path.join(run_dir, f"{exp_id}_ep{epoch}.pth")
        logger.info('saving model to %s...', save_path)
        save_dict["epoch"] = epoch
        save_dict["global_step"] = global_step
        try:
            save_dict["model_state_dict"] = model.module.state_dict()
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
            save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, save_path)
        except Exception as e:
            logger.error("Failed to save checkpoint! Error: %s", e)

        if (epoch + 1) % sample_interval == 0:
            try:
                all_masks = all_masks_by_obs[obs]
                all_generation_idx = all_generation_idx_by_obs[obs]
                sample_order_i = np.random.randint(len(all_masks))

                batch_to_complete = get_sampling_images(test_loader)

                logger.info('sampling images with observation ordering variant %d...', sample_order_i)
                sample_t = sample(model,
                                  all_generation_idx[sample_order_i],
                                  *all_masks[sample_order_i],
                                  batch_to_complete,
                                  obs)
                sample_save_path = os.path.join(run_dir, f"tsample_obs_{epoch}_order{sample_order_i}.png")
                utils.save_image(sample_t, sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)
            except Exception as e:
                logger.error("Failed to sample images! Error: %s", e)
