from __future__ import annotations
import logging
import time
from collections import defaultdict
from warnings import simplefilter
import numpy as np
import torch
import torch.utils.data
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from torchmetrics.regression import MeanSquaredError, R2Score, PearsonCorrCoef
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from typing import List, Optional


# Silence repeated convergence warnings from scikit-learn Ridge regression.
simplefilter("ignore", category=ConvergenceWarning)


class LinearProbingRegCallback(Callback):
    """Linear probing on top of Multi-Modal models for regression."""
    def __init__(self, downstream_data_modules: List[LightningDataModule],
                 names: Optional[List[str]] = None,
                 val_loaders: bool = True,
                 use_sklearn: bool = True,
                 frequency: str = "by_epoch",
                 logging_level: str = "INFO",
                 **extraction_kwargs):
        """
        :param downstream_data_modules: List of dataset to evaluate
        :param names: Names of each dataset
        :param val_loaders: If True, use validation set for validation
        :param multilabel: If True, linear probes are fitted for each target label
            and acc@1/F1-score are reported instead of acc@1/acc@5
        :param use_sklearn: if True, use scikit-learn estimators for linear probing (on CPU).
            Otherwise, torch estimators are fitted (on CPU or GPU).
        :param frequency: {'by_epoch', 'by_fit'}
        :param logging_level: in {'INFO', 'DEBUG', 'WARNING', 'CRITICAL', 'ERROR'}
        :param extraction_kwargs: Keyword arguments given to `extract_features` method
        """
        self.downstream_data_modules = downstream_data_modules
        self.names = names
        self.val_loaders = val_loaders
        self.use_sklearn = use_sklearn
        self.frequency = frequency
        if not self.use_sklearn:
            raise NotImplementedError("Ridge regression not implemented with PyTorch.")
        self.logging_level = logging_level
        self.extraction_kwargs = extraction_kwargs
        if self.names is None:
            self.names = [d.test_dataloader().__class__.__name__ for d in downstream_data_modules]

    def linear_probing(self,  trainer: Trainer, pl_module: LightningModule, log: bool = True):
        if trainer.global_rank == 0:
            if not hasattr(pl_module, "extract_features"):
                raise ValueError("`extract_features` must be implemented for linear probing")
            scores = defaultdict(list)
            for downstream_data_mod, dataset in zip(self.downstream_data_modules, self.names):
                train_loader, val_loader, test_loader = (
                    downstream_data_mod.train_dataloader(),
                    downstream_data_mod.val_dataloader(),
                    downstream_data_mod.test_dataloader())
                train_features, train_labels = pl_module.extract_features(train_loader, **self.extraction_kwargs)
                test_features, test_labels = pl_module.extract_features(test_loader, **self.extraction_kwargs)
                val_features, val_labels = None, None
                if self.val_loaders:
                    val_features, val_labels = pl_module.extract_features(val_loader, **self.extraction_kwargs)
                scores_ = (
                    evaluate_linear_probe(train_features, train_labels, test_features,
                                          test_labels, val_features, val_labels,
                                          use_sklearn=self.use_sklearn,
                                          logging_level=self.logging_level))
                for k, v in scores_.items():
                    scores[k].append(v)
                print('Linear probe - regression ({d}): {scores}'
                      .format(d=dataset, scores="  ".join(map(lambda k: "%s=%.3f"%(k, scores_[k]), scores_))))

            for k, v in list(scores.items()):
                if len(self.names) > 1:
                    for i, dataset in enumerate(self.names):
                        scores["%s_%s"%(k, dataset)] = scores[k][i]
                scores[k] = np.mean(scores[k])

            if log:
                pl_module.log_dict(dict(scores), on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.frequency == "by_epoch":
            self.linear_probing(trainer, pl_module)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.frequency == "by_fit":
            self.linear_probing(trainer, pl_module, log=False)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self.linear_probing(trainer, pl_module)


def evaluate_linear_probe(
    train_feats,
    train_labels,
    test_feats,
    test_labels,
    val_feats=None,
    val_labels=None,
    holdout_fraction=0.8,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=True,
    logging_level="INFO"
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic.
    """
    start = time.time()
    logger = logging.getLogger("Linear probing")
    logger.setLevel(logging_level)
    if val_feats is None or val_labels is None:
        train_idx, val_idx = next(ShuffleSplit(train_size=holdout_fraction, random_state=48).
                                  split(np.ones((len(train_feats), 1))))
        val_feats = train_feats[val_idx]
        val_labels = train_labels[val_idx]
        train_feats = train_feats[train_idx]
        train_labels = train_labels[train_idx]

    regressor = train_linear_probe(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        use_sklearn=use_sklearn,
        logger=logger
    )
    scores = test_linear_probe(regressor, test_feats, test_labels, use_sklearn=use_sklearn, logger=logger)

    del regressor
    torch.cuda.empty_cache()
    logger.debug(f"Time taken {time.time() - start:.2f}")
    return scores


def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    max_iter=None,
    combine_trainval=True,
    use_sklearn=False,
    logger = None
):
    if logger is None:
        logger = logging.getLogger("Linear probing")
    device = train_feats.device
    num_outputs = train_labels.shape[-1]
    score_meter = MeanSquaredError(num_outputs=num_outputs).to(device)

    # CLIP performed linear probe evaluation by sweeping over 96 log-spaced costs.
    # For simplicity, we sweep in one coarse stage for quick search.
    costs = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
    logger.debug(f"First sweep with costs: {costs}")

    # Train and avaluate each classifier and get accuracy.
    scores = []
    for cost in costs:
        regressor = _fit_reg(
            train_feats, train_labels, cost, max_iter, use_sklearn
        )
        if use_sklearn:
            predictions = torch.from_numpy(regressor.predict(valid_feats.cpu().numpy())).to(device)
        else:
            predictions = regressor.predict(valid_feats)
        score = score_meter(predictions, valid_labels)
        if score.ndim > 0: # for multioutput
            score = score.sum()
        scores.append(score)

        score_meter.reset()
        logger.debug(f"Cost = {cost}, MSE = {score:.3f}")

    best_score = min(scores)
    best_cost = costs[scores.index(best_score)]

    logger.debug(f"Best cost = {best_cost:.3f}, MSE = {best_score:.3f}")

    # train final classifier
    if combine_trainval:
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)

        final_regressor = _fit_reg(
            trainval_feats,
            trainval_labels,
            best_cost,
            max_iter,
            use_sklearn
        )
    else:
        final_regressor = _fit_reg(
            train_feats, train_labels, best_cost, max_iter, use_sklearn
        )

    return final_regressor


def test_linear_probe(
    linear_regressor, test_feats, test_labels, use_sklearn=False, logger=None
):
    if logger is None:
        logger = logging.getLogger("Linear probing")
    # evaluate
    device = test_feats.device
    num_outputs = test_labels.shape[-1]
    mse = MeanSquaredError(num_outputs=num_outputs).to(device)
    R2 = R2Score(num_outputs=num_outputs, multioutput="uniform_average").to(device)
    pearson_r = PearsonCorrCoef(num_outputs=num_outputs).to(device)
    if use_sklearn:
        predictions = torch.as_tensor(linear_regressor.predict(test_feats.cpu().numpy())).to(device)
    else:
        predictions = linear_regressor.predict(test_feats)

    mse_ = mse(torch.as_tensor(predictions), test_labels)
    R2_ = float(R2(torch.as_tensor(predictions), test_labels))
    pearson_r_ = pearson_r(torch.as_tensor(predictions), test_labels)
    if pearson_r_.ndim > 0:
        pearson_r_ = pearson_r_.mean()
    if mse_.ndim > 0:
        mse_ = mse_.mean()
    pearson_r_ = float(pearson_r_)
    mse_ = float(mse_)
    logger.info(f"Test MSE/R2/Pearson-r: {mse_:.5f}/{R2_:.3f}/{pearson_r_:.3f}")
    return {"mse": mse_, "r2": R2_, "pearson_r": pearson_r_}


def _fit_reg(
    feats: torch.Tensor,
    labels: torch.Tensor,
    cost: float,
    max_iter: int = 100,
    use_sklearn: bool = False,
) -> Ridge:
    """
    Initialize and fit a `LogisticRegression` classifier for input features and
    labels. Default settings follow CLIP (L-BFGS, 1K iterations, etc.).
    """
    if use_sklearn:
        regressor = Ridge(
            alpha=cost, max_iter=max_iter
        )
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        raise NotImplementedError()
    regressor.fit(feats, labels)
    return regressor

