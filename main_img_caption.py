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
from pytorch_lightning.loggers import TensorBoardLogger
from evaluation.linear_probe import LinearProbingCallback
from utils import CheckNaNGradCallback
from dataset.img_caption import DownstreamVisionDataModule


@hydra.main(version_base=None, config_name="train_img_caption", config_path="./configs")
def main(cfg: DictConfig):
    # fix the seed for repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # create model + save hyper-parameters
    dataset = cfg.data.data_module.dataset  # Which Image-Caption dataset to load
    model_kwargs = dict()
    if cfg.model.name == "CoMM":  # Define encoders + adapters for MMFusion
        encoders = instantiate(cfg[dataset]["encoders"])  # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"])  # adapters also specific
        model_kwargs = dict(encoder=dict(encoders=encoders, input_adapters=adapters))

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **model_kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    linear_eval_kwargs = dict()
    if cfg.model.name == "CoMM": # Specify which modality to test
        linear_eval_kwargs = dict(mask_modalities=[m == "vision" for m in cfg[dataset]["modalities"]])
    elif cfg.model.name == "CLIP":
        linear_eval_kwargs["encode_text"] = False

    callbacks = [
        LinearProbingCallback([
            DownstreamVisionDataModule(d, cfg.data.data_module.batch_size, cfg.data.data_module.num_workers)
            for d in cfg.downstream_datasets],
            logging_level="INFO",
            names=cfg.downstream_datasets,
            val_loaders=False,  # Not official validation split for most of downstream dataset
            **linear_eval_kwargs),
        CheckNaNGradCallback(stop_if_nan=False)
    ]

    # Trainer + fit / eval
    trainer = instantiate(
        cfg.trainer,
        default_root_dir = build_root_dir(cfg),
        logger=TensorBoardLogger(build_root_dir(cfg), name="logs"),
        callbacks=callbacks)

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, cfg.data.data_module.dataset)

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