from hydrotail.models.sequence_tail import SequenceTailModel
from hydrotail.models.sklearn_tail import GBDTTailModel, LinearTailModel
from hydrotail.models.torch_tail import TorchTailModel

__all__ = ["GBDTTailModel", "LinearTailModel", "TorchTailModel", "SequenceTailModel"]
