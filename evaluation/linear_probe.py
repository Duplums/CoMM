# Based on evaluate_zeroshot from SLIP but changed by MB
from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from math import log10
from warnings import simplefilter
import numpy as np
import torch
import torch.utils.data
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score
from evaluation.logistic_regression import LogisticRegression
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy, F1Score, AUROC
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from typing import List, Optional


# Silence repeated convergence warnings from scikit-learn logistic regression.
simplefilter("ignore", category=ConvergenceWarning)


class LinearProbingCallback(Callback):
    """Linear probing on top of Multi-Modal models."""
    def __init__(self, downstream_data_modules: List[LightningDataModule],
                 names: Optional[List[str]] = None,
                 val_loaders: bool = True,
                 multilabel: bool = False,
                 use_sklearn: bool = False,
                 fastsearch: bool = False,
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
        :param fastsearch: if True, perform coarse grid-search of costs in linear
            probing (faster but sub-optimal)
        :param frequency: {'by_epoch', 'by_fit'}
        :param logging_level: in {'INFO', 'DEBUG', 'WARNING', 'CRITICAL', 'ERROR'}
        :param extraction_kwargs: Keyword arguments given to `extract_features` method
        """
        self.downstream_data_modules = downstream_data_modules
        self.names = names
        self.val_loaders = val_loaders
        self.multilabel = multilabel
        self.use_sklearn = use_sklearn
        self.fastsearch = fastsearch
        self.frequency = frequency
        if self.multilabel and not self.use_sklearn:
            raise NotImplementedError("`multilabel` linear probing not implemented with PyTorch.")
        self.logging_level = logging_level
        self.extraction_kwargs = extraction_kwargs
        if self.names is None:
            self.names = [d.test_dataloader().__class__.__name__ for d in downstream_data_modules]

    def linear_probing(self,  trainer: Trainer, pl_module: LightningModule, log: bool = True):
        if trainer.global_rank == 0:
            if not hasattr(pl_module, "extract_features"):
                raise ValueError("`extract_features` must be implemented for linear probing")
            scores = defaultdict(list)
            if self.multilabel:
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
                                          multilabel=self.multilabel, use_sklearn=self.use_sklearn,
                                          fastsearch=self.fastsearch, logging_level=self.logging_level))
                for k, v in scores_.items():
                    scores[k].append(v)
                print('Linear probe ({d}): {scores}'
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
    multilabel=False,
    holdout_fraction=0.6,
    use_mean_accuracy=True,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
    fastsearch=False,
    logging_level="INFO"
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    start = time.time()
    logger = logging.getLogger("Linear probing")
    logger.setLevel(logging_level)
    if val_feats is None or val_labels is None:
        if multilabel: # simplify the problem
            train_idx, val_idx = next(ShuffleSplit(train_size=holdout_fraction, random_state=48).
                                      split(np.ones((len(train_feats), 1))))
        else:
            train_idx, val_idx = next(StratifiedShuffleSplit(train_size=holdout_fraction, random_state=48).
                                      split(np.ones((len(train_feats), 1)), train_labels.cpu().numpy()))
        val_feats = train_feats[val_idx]
        val_labels = train_labels[val_idx]
        train_feats = train_feats[train_idx]
        train_labels = train_labels[train_idx]

    classifier = train_linear_probe(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        use_mean_accuracy,
        sk_verbose,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        use_sklearn=use_sklearn,
        multilabel=multilabel,
        fastsearch=fastsearch,
        logger=logger
    )
    test_acc = test_linear_probe(classifier, test_feats, test_labels, use_mean_accuracy,
                                 use_sklearn=use_sklearn, multilabel=multilabel, logger=logger)

    del classifier
    torch.cuda.empty_cache()
    logger.debug(f"Time taken {time.time() - start:.2f}")
    return test_acc


def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    use_mean_accuracy=True,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
    multilabel=False,
    fastsearch=False,
    logger = None
):
    if logger is None:
        logger = logging.getLogger("Linear probing")
    device = train_feats.device
    average = "macro" if use_mean_accuracy else "micro"
    if multilabel:
        num_labels = train_labels.shape[-1]
        acc_meter = F1Score(task="multilabel", num_labels=num_labels, average=average).to(device)
    else:
        NUM_C = len(train_labels.unique())
        acc_meter = MulticlassAccuracy(
            num_classes=NUM_C,
            average=average).to(device)

    # CLIP performed linear probe evaluation by sweeping over 96 log-spaced costs.
    # Following CLIP, we sweep in two stages (coarse and fine) for quick search.
    costs = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
    logger.debug(f"First sweep with costs: {costs}")

    # Train and avaluate each classifier and get accuracy.
    accuracies = []
    for cost in costs:
        classifier = _fit_logreg(
            train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn, multilabel
        )
        if use_sklearn:
            predictions = torch.from_numpy(classifier.predict(valid_feats.cpu().numpy())).to(device)
        else:
            predictions = classifier.predict_proba(valid_feats)
        accuracy = acc_meter(predictions, valid_labels)
        accuracies.append(accuracy)

        acc_meter.reset()
        logger.debug(f"Cost = {cost}, Top-1 accuracy = {accuracy:.3f}")

    best_accuracy = max(accuracies)
    best_cost = costs[accuracies.index(best_accuracy)]

    if not fastsearch:
        # Second sweep: search around the best cost with a resolution of 8 steps per
        # decade. Example: if best cost = 1e2, then these costs will be in (1, 1e-4).
        costs = torch.logspace(log10(best_cost) - 2, log10(best_cost) + 2, 29)
        costs = costs[(costs >= 1e-6) & (costs <= 1e6)].tolist()

        # We may visit the same cost value multiple times while searching, to avoid
        # re-training the classifier, keep a map of accuracies per cost.
        accuracies = {best_cost: best_accuracy}

        logger.debug("Performing second sweep as a binary search around best cost.")
        logger.debug(f"Initial search space: {[round(c, 3) for c in costs]}")

        while len(costs) > 1:
            # Get mid-points of left/right half interval of search space: (25,50,75)%
            cost_25 = costs[len(costs) // 4]
            cost_50 = costs[len(costs) // 2]
            cost_75 = costs[-len(costs) // 4]
            logger.debug(
                f"Half interval mid-points: {cost_25=:.3f}, {cost_50=:.3f}, {cost_75=:.3f}"
            )

            # Compute accuracy for these costs (skip if computed in prev iteration).
            for cost in [cost_25, cost_50, cost_75]:
                _acc = accuracies.get(cost, None)
                if _acc is None:
                    classifier = _fit_logreg(
                        train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn, multilabel
                    )
                    if use_sklearn:
                        predictions = classifier.predict(valid_feats.cpu().numpy())
                        _acc = acc_meter(torch.from_numpy(predictions).to(device), valid_labels)
                    else:
                        predictions = classifier.predict_proba(valid_feats)
                        _acc = acc_meter(predictions, valid_labels)
                    accuracies[cost] = _acc
                    acc_meter.reset()

                logger.debug(f"Cost = {round(cost, 3)}, Top-1 accuracy = {_acc:.3f}")

            # Cut down the search space by half such that the mid-point of the resulting
            # reduced search space is the cost with current best accuracy.
            max_acc = max(accuracies[cost_25], accuracies[cost_50], accuracies[cost_75])
            costs = (
                costs[: len(costs) // 2]
                if max_acc == accuracies[cost_25]
                else costs[len(costs) // 2 :]
                if max_acc == accuracies[cost_75]
                else costs[len(costs) // 4 : -len(costs) // 4]
            )
            logger.debug(f"Reduced search space, costs: {[round(c, 3) for c in costs]}")

        # Filter None accuracy values (some costs may not be visited while searching).
        # Then find best accuracy and its cost.
        best_cost, best_accuracy = max(accuracies.items(), key=lambda k: k[1])

    logger.debug(f"Best cost = {best_cost:.3f}, Top-1 accuracy = {best_accuracy:.3f}")

    # train final classifier
    if combine_trainval:
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)

        final_classifier = _fit_logreg(
            trainval_feats,
            trainval_labels,
            best_cost,
            sk_verbose,
            max_iter,
            use_sklearn,
            multilabel
        )
    else:
        final_classifier = _fit_logreg(
            train_feats, train_labels, best_cost, sk_verbose, max_iter, use_sklearn, multilabel
        )

    return final_classifier


def test_linear_probe(
    linear_classifier, test_feats, test_labels, use_mean_accuracy,
        use_sklearn=False, num_classes=None, multilabel=False, logger=None
):
    if logger is None:
        logger = logging.getLogger("Linear probing")
    # evaluate
    average = "macro" if use_mean_accuracy else "micro"
    device = test_feats.device
    if multilabel:
        num_labels = test_labels.shape[-1]
        acc1 = MultilabelAccuracy(num_labels=num_labels, average=average).to(device)
        f1_score = F1Score(task="multilabel", num_labels=num_labels, average=average).to(device)
        f1_weighted_score = F1Score(task="multilabel", num_labels=num_labels, average="weighted").to(device)
        if use_sklearn:
            predictions = torch.as_tensor(linear_classifier.predict(test_feats.cpu().numpy())).to(device)
        else:
            predictions = linear_classifier.predict(test_feats)
        accuracy1 = float(acc1(predictions, test_labels))
        f1_mean = float(f1_score(predictions, test_labels))
        f1_weighted = float(f1_weighted_score(predictions, test_labels))
        acc1_sklearn = accuracy_score(test_labels.cpu().numpy(), predictions.cpu().numpy())
        logger.info(f"Test acc@1/acc@1(subset)/f1-score/f1-weighted: {accuracy1:.3f}/{acc1_sklearn:.3f}/{f1_mean:.3f}/{f1_weighted:.3f}")
        return {"acc1": accuracy1, "f1_mean": f1_mean, "f1_weighted": f1_weighted, "acc1(subset)": acc1_sklearn}
    else:
        NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
        acc1 = MulticlassAccuracy(num_classes=NUM_C, average=average).to(test_feats.device)
        acc_per_class = MulticlassAccuracy(num_classes=NUM_C, average=None).to(test_feats.device)
        acc5 = MulticlassAccuracy(num_classes=NUM_C, average=average,
                                  top_k=min(NUM_C, 5)).to(test_feats.device)
        roc_auc = AUROC(task="multiclass", num_classes=NUM_C).to(test_feats.device)
        if use_sklearn:
            predictions = torch.as_tensor(linear_classifier.predict_proba(test_feats.cpu().numpy())).to(device)
        else:
            predictions = linear_classifier.predict_proba(test_feats)
        accuracy1 = float(acc1(torch.as_tensor(predictions), test_labels))
        accuracy5 = float(acc5(torch.as_tensor(predictions), test_labels))
        accuracy_per_class = [float(x) for x in acc_per_class(torch.as_tensor(predictions), test_labels)]
        auc = float(roc_auc(torch.as_tensor(predictions), test_labels))
        logger.info(f"Test acc@1/acc@5/acc_per_class/roc_auc: {accuracy1:.3f}/{accuracy5:.3f}/{accuracy_per_class}/{auc}")
        return {"acc1": accuracy1, "acc5": accuracy5, "roc_auc": auc}

def _fit_logreg(
    feats: torch.Tensor,
    labels: torch.Tensor,
    cost: float,
    verbose: bool = False,
    max_iter: int = 100,
    use_sklearn: bool = False,
    multilabel : bool = False
) -> LogisticRegression:
    """
    Initialize and fit a `LogisticRegression` classifier for input features and
    labels. Default settings follow CLIP (L-BFGS, 1K iterations, etc.).
    """
    if use_sklearn:
        classifier = sk_LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose
        )
        if multilabel:
            classifier = MultiOutputClassifier(classifier)
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        classifier = LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose
        )
        if multilabel:
            raise NotImplementedError()
    classifier.fit(feats, labels)
    return classifier


def split_trainval(targets, val_percentage):
    # Organize dataset by classes (class ID -> list[dataset index] map).
    labels_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        labels_to_indices[label].append(index)

    train_indices = []
    valid_indices = []
    for label, indices in labels_to_indices.items():
        # Deterministic shuffling to ensure same held-out split across runs.
        random.Random(93).shuffle(indices)

        train_indices.extend(indices[int(len(indices) * val_percentage) :])
        valid_indices.extend(indices[: int(len(indices) * val_percentage)])

    return train_indices, valid_indices