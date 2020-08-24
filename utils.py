import logging

import numpy as np
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("gen")


def configure_logger(filename="debug.log"):
    logger = logging.getLogger("gen")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s|%(name)s|%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis), inplace=True)


###########################
# Shared loss utilities
###########################

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def average_loss(log_probs_fn, x, ls, *xargs):
    """ ensemble multiple nn outputs (ls) by averaging likelihood """
    # Ensembles at the level of the joint distribution
    all_log_probs = []
    for l in ls:
        log_probs = log_probs_fn(x, l, *xargs)  # B x H x W x num_logistic_mix
        log_prob = log_sum_exp(log_probs)  # B x H x W
        log_prob = torch.sum(log_prob, dim=(1, 2))  # B, log prob of image under this
                                                    # ensemble component
        all_log_probs.append(log_prob)
    all_log_probs = torch.stack(all_log_probs, dim=1) - np.log(len(ls))  # B x len(ls)
    loss = -torch.sum(log_sum_exp(all_log_probs))
    return loss

######################################################
# Binarization utilities and cross entropy losses
######################################################

def _binarized_label(x):
    assert x.size(1) == 1
    x = x * .5 + .5  # Scale from [-1, 1] to [0, 1] range
    x = binarize_torch(x)  # binarize image. Should be able to just cast,
                            # since x is either 0. or 1., but this could avoid float
                            # innacuracies from rescaling.
    x = x.squeeze(1).long()
    return x


def _binarized_log_probs(x, l):
    """Cross-entropy loss
    Args:
        x: B x H x W floating point ground truth image, [-1, 1] scale
        l: B x 2 x H x W output of neural network
    Returns:
        log_probs: B x H x W x 1 tensor of likelihod of each pixel in x
    """
    assert l.size(1) == 2
    x = _binarized_label(x)
    l = F.log_softmax(l, dim=1)
    log_probs = -F.nll_loss(l, x, reduction="none").unsqueeze(-1)
    return log_probs


def binarized_loss(x, l):
    """Cross-entropy loss
    Args:
        x: B x 1 x H x W floating point ground truth image, [-1, 1] scale
        l: B x 2 x H x W output of neural network
    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    # cross_entropy averages across the batch, so we multiply by batch size
    # to keep a similar loss scale as with grayscale MNIST
    return F.cross_entropy(l, x, reduction="sum")


def binarized_loss_averaged(x, ls):
    """
    Args:
        x: B x C x H x W ground truth image
        ls: list of B x 2 x H x W outputs of NN
    Returns:
        loss: 0-dimensional NLL loss tensor
    """
    return average_loss(_binarized_log_probs, x, ls)


def binarize_np(images: np.ndarray):
    rand = np.random.uniform(size=images.shape)
    return (rand < images).astype(np.float32)


def binarize_torch(images):
    rand = torch.rand(images.shape, device=images.device)
    return (rand < images).float()

###################
# 1D (1 color) loss
###################

def discretized_mix_logistic_log_probs_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return log_probs


def discretized_mix_logistic_loss_1d(x, l):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    
    Args:
        x: B x C x H x W ground truth image
        l: B x (3 * num_logistic_mix) x H x W output of NN
    """
    log_probs = discretized_mix_logistic_log_probs_1d(x, l)
    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d_averaged(x, ls):
    """ reduced (summed) log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    Averages likelihood across multiple sets of mixture parameters
    
    Args:
        x: B x C x H x W ground truth image
        ls: list of B x (3 * num_logistic_mix) x H x W outputs of NN
    """
    return average_loss(discretized_mix_logistic_log_probs_1d, x, ls)

###########
# Sampling
###########

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, coord1, coord2, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out.data[:, :, coord1, coord2]


#########################################################################################
# utilities for shifting the image around, efficient alternative to masking convolutions
#########################################################################################

def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed 
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed 
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


#######################
# Restoring checkpoint
#######################

def load_part_of_model(path, model, optimizer=None):
    checkpoint = torch.load(path)
    params = checkpoint["model_state_dict"]
    # Restore model
    logger.info("Restoring model from %s", path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                logger.warning("Error loading model.state_dict()[%s]: %s", name, e)
        else:
            logger.warning("Key present in checkpoint that is not present in model.state_dict(): %s", name)
    logger.info('Loadded %s fraction of params:' % (added / float(len(model.state_dict().keys()))))

    # Restore optimizer
    if optimizer:
        logger.info("Restoring optimizer from %s", path)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info('Loaded optimizer params directly')
        except Exception as e:
            logger.warning("Failed to load entire optimizer state dict at once, trying each key of state only")

            added = 0
            for name, param in checkpoint["optimizer_state_dict"]["state"].items():
                if name in optimizer.state_dict()["state"].keys():
                    try:
                        optimizer.state_dict()["state"][name].copy_(param)
                        added += 1
                    except Exception as e:
                        logger.error("Error loading optimizer.state_dict()['state'][%s]: %s", name, e)
                        pass
            logger.info('Loaded %s fraction of optimizer params:' % (added / float(len(optimizer.state_dict()["state"].keys()))))

            # TODO: load param_groups key?

    return checkpoint["epoch"], checkpoint.get("global_step", -1)


class EMA():
    # Computes exponential moving average of model parameters, adapted from https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, model):
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone()

    def update(self, model):
        for name, param in model.state_dict().items():
            assert name in self.shadow
            new_average = self.mu * param + (1.0 - self.mu) * self.shadow[name]
            self.shadow[name] = new_average.clone()
            return new_average

    def state_dict(self):
        return self.shadow
