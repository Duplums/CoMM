from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from utils import CheckNaNGradCallback

@hydra.main(version_base=None, config_name="train_hateful_memes", config_path="./configs")
def main(cfg: DictConfig):
    """
    Training/test of Multi-Modal models on Hateful Memes dataset.
    Models currently implemented are:
        - CoMM [ours!]
        - SimCLR
        - CLIP
    """

    # fix the seed for repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # create model + save hyper-parameters
    dataset = "hateful_memes"
    model_kwargs = dict()
    if cfg.model.name == "CoMM":  # Define encoders + adapters for MMFusion
        encoders = instantiate(cfg[dataset]["encoders"])  # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"])  # adapters also specific
        model_kwargs = dict(encoder=dict(encoders=encoders, input_adapters=adapters))

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **model_kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    downstream_data_module = instantiate(cfg.data.data_module, model="Sup")

    logger = TensorBoardLogger(build_root_dir(cfg), name="logs")
    callbacks = [instantiate(cfg.linear_probing, downstream_data_modules=[downstream_data_module], names=[dataset]),
                 ModelCheckpoint(monitor='roc_auc', mode='max', save_top_k=1)]

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir = build_root_dir(cfg),
        logger=logger,
        callbacks=callbacks)

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "hateful_memes")

    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        if getattr(cfg, "ckpt_path", None) is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == '__main__':
    main()