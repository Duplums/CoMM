import numpy as np
import os
import glob
import re
import shutil
import torch
import random
from PIL import ImageFilter
import torch.distributed as dist
import torch.autograd as autograd
from omegaconf import DictConfig
from typing import Tuple, List, Optional
from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning import Callback
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import warnings
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TextMasking(torch.nn.Module):
    """
        Randomly mask input tokens using a special `mask` token.
    """
    def __init__(self, mask_prob: float, mask_token_id: int, mask_ignored_ids: Optional[List[int]] = None) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.mask_ignored_ids = mask_ignored_ids or [] # ignore these tokens for masking

    def _init_full_mask(self, seq: torch.Tensor) -> torch.Tensor:
        # Returns `True` for tokens to not ignore in `seq`
        full_mask = torch.full_like(seq, True, dtype=torch.bool)
        for ignored_id in self.mask_ignored_ids:
            full_mask &= (seq != ignored_id)
        return full_mask

    def _get_mask_subset_with_prob(self, mask: torch.Tensor) -> torch.Tensor:
        # Returns a subset of input `mask`
        random_mask = torch.rand(mask.shape, device=mask.device) < self.mask_prob
        mask &= random_mask
        return mask

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob == 0:
            return seq
        else:
            mask = self._init_full_mask(seq)
            mask = self._get_mask_subset_with_prob(mask)
            masked_seq = seq.clone().detach()
            masked_seq.masked_fill_(mask, self.mask_token_id)
            return masked_seq


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CheckNaNGradCallback(Callback):
    def __init__(self, stop_if_nan: bool = True):
        """
        :param stop_if_nan: whether to stop training or not if NaN gradient is found
        """
        self.stop_if_nan = stop_if_nan

    def on_after_backward(self, trainer, pl_module):
        should_stop = False
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradients found in parameter '{name}'")
                should_stop = True
        if should_stop and not self.stop_if_nan:
            # reset gradients for this batch
            pl_module.zero_grad()
        trainer.should_stop = (should_stop & self.stop_if_nan)


def set_weight_decay_per_param(
        model: torch.nn.Module,
        weight_decay: float):
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    optim_params = [{"params": p_wd, "weight_decay": weight_decay},
                    {"params": p_non_wd, "weight_decay": 0}]

    return optim_params


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dirs(dirs):
    # dirs is a list
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def save_on_master(state, is_best, output_dir, epoch: int=None):
    if is_main_process():
        ckpt_path = f'{output_dir}/checkpoint.pt'
        if epoch is not None:
            ckpt_path = f'{output_dir}/checkpoint_{epoch}.pt'
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(cfg: DictConfig):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        cfg.distributed = False
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        cfg.rank, cfg.dist_url), flush=True)
    torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)


def parse_tensorboard(path: str, scalars: Optional[List[str]] = None):
    """Parse an events file from Tensorboard and
        returns a dictionary of pandas dataframes for all scalars"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1}
    )
    _absorb_print = ea.Reload() # laods event from File
    if scalars is not None:
        # make sure the scalars are in the event accumulator tags
        assert all(
            s in ea.Tags()["scalars"] for s in scalars
        ), f"some scalars were not found in the event accumulator {path}"
        return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    return {k: pd.DataFrame(ea.Scalars(k)) for k in ea.Tags()["scalars"]}


def resume_checkpoint(cfg: DictConfig, optimizer, model):
    best_acc1 = 0.
    if cfg.paths.resume != "":
        if os.path.isfile(cfg.paths.resume):
            print("=> loading resume checkpoint '{}'".format(cfg.paths.resume))
            checkpoint = torch.load(cfg.paths.resume, map_location='cuda')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            cfg.optim.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(cfg.paths.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.paths.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(cfg.paths.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cuda')
            cfg.optim.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
    return best_acc1


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    # TODO: update the LR at each iteration (not epoch)
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> from torch import nn
        >>> from torch.optim import Adam
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

