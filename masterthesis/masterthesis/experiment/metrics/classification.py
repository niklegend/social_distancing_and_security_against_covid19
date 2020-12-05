import math
from typing import Optional

import torch
from sklearn.metrics import confusion_matrix as skl_confusion_matrix

from .metric import Metric


class _Accuracy(Metric):

    def __init__(self, name):
        super(_Accuracy, self).__init__(name)

        self.corrects = 0
        self.count = 0

    def reset(self) -> None:
        self.corrects = 0
        self.count = 0

    def update(self, targets: torch.Tensor, outputs: torch.Tensor) -> None:
        predictions = self.as_predictions(outputs)

        self.count += targets.size(0)
        self.corrects += torch.sum(predictions == targets).item()

    def result(self) -> float:
        return self.corrects / self.count

    def as_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Accuracy(_Accuracy):

    def __init__(self, name: Optional[str] = 'accuracy'):
        super(Accuracy, self).__init__(name)

    def as_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs.argmax(dim=1)


class BinaryAccuracy(_Accuracy):

    def __init__(self, name: Optional[str] = 'accuracy', threshold: Optional[float] = 0.5):
        super(BinaryAccuracy, self).__init__(name)

        self.threshold = threshold

    def as_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs > self.threshold


def binary_confusion_matrix(targets, predictions):
    return skl_confusion_matrix(targets.numpy(), predictions.numpy()).ravel()


class Precision(Metric):

    def __init__(self, name: Optional[str] = 'precision', threshold: Optional[float] = 0.5):
        super(Precision, self).__init__(name)

        self.threshold = threshold

        self.true_positives = 0
        self.false_positives = 0

    def reset(self) -> None:
        self.true_positives = 0
        self.false_positives = 0

    def update(self, targets: torch.Tensor, outputs: torch.Tensor) -> None:
        predictions = outputs > self.threshold
        _, fp, _, tp = binary_confusion_matrix(targets, predictions)
        # tn, fp, fn, tp

        self.true_positives += tp
        self.false_positives += fp

    def result(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives)


class Recall(Metric):

    def __init__(self, name: Optional[str] = 'recall', threshold: Optional[float] = 0.5):
        super(Recall, self).__init__(name)

        self.threshold = threshold

        self.true_positives = 0
        self.false_negatives = 0

    def reset(self) -> None:
        self.true_positives = 0
        self.false_negatives = 0

    def update(self, targets: torch.Tensor, outputs: torch.Tensor) -> None:
        predictions = outputs > self.threshold
        _, _, fn, tp = binary_confusion_matrix(targets, predictions)
        # tn, fp, fn, tp

        self.true_positives += tp
        self.false_negatives += fn

    def result(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives)


class BinaryConfusionMatrix(Metric):

    @staticmethod
    def from_values(true_positives, false_positives, false_negatives, true_negatives):
        cm = BinaryConfusionMatrix()

        cm.true_positives = true_positives
        cm.false_positives = false_positives
        cm.false_negatives = false_negatives
        cm.true_negatives = true_negatives

        return cm

    def __init__(self, threshold: Optional[float] = 0.5):
        super(BinaryConfusionMatrix, self).__init__(None)

        self.threshold = threshold

        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def reset(self) -> None:
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, targets: torch.Tensor, outputs: torch.Tensor) -> None:
        predictions = outputs > self.threshold
        tn, fp, fn, tp = binary_confusion_matrix(targets, predictions)

        self.true_negatives += tn
        self.false_positives += fp
        self.false_negatives += fn
        self.true_positives += tp

    def result(self) -> float:
        return self.phi_coefficient()

    def as_dict(self, tag: Optional[str] = None) -> dict:
        return {
            Metric.key('true_positives', tag): self.true_positives,
            Metric.key('false_positives', tag): self.false_positives,
            Metric.key('false_negatives', tag): self.false_negatives,
            Metric.key('true_negatives', tag): self.true_negatives,
        }

    def to_str(self, tag: Optional[str] = None) -> str:
        return ', '.join([
            Metric.str('phi_coefficient', self.phi_coefficient(), tag),
            Metric.str('precision', self.precision(), tag),
            Metric.str('recall', self.recall(), tag),
        ])

    @property
    def positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def negatives(self) -> int:
        return self.true_negatives + self.false_positives

    def recall(self) -> float:
        # True Positive Rate
        return self.true_positives / self.positives

    def precision(self) -> float:
        # Positive Predictive Value
        return self.true_positives / (self.true_positives + self.false_positives)

    def accuracy(self, balanced: Optional[bool] = False) -> float:
        if balanced:
            return (self.recall() + self.selectivity()) * 0.5
        return (self.true_positives + self.true_negatives) / (self.positives + self.negatives)

    def f_score(self, beta: Optional[float] = 1.0) -> float:
        assert beta != 0
        beta_sqr = beta * beta

        ppv = self.precision()
        tpr = self.recall()

        return ((1 + beta_sqr) * ppv * tpr) / (beta_sqr * ppv + tpr)

    def phi_coefficient(self):
        num = self.true_positives * self.true_negatives - self.false_positives * self.false_negatives
        den = 1
        for t in [self.true_positives, self.true_negatives]:
            for f in [self.false_positives, self.false_negatives]:
                den *= t + f
        return num / math.sqrt(den)

    def selectivity(self) -> float:
        # True Negative Rate
        return self.true_negatives / self.negatives

    def prevalence_threshold(self) -> float:
        tpr = self.recall()
        tnr = self.selectivity()
        return (math.sqrt(tpr * (1 - tnr)) + tnr - 1) / (tpr + tnr - 1)

    def threat_score(self) -> float:
        return self.true_positives / (self.positives + self.false_positives)

    def informedness(self) -> float:
        return self.recall() + self.selectivity() - 1
