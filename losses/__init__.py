from .CrossEntropyLoss import CrossEntropyLoss
from .FocalLoss import FocalLoss
from .GradientHarmonizedLoss import GradientHarmonizedLoss

loss_names = {
    'CrossEntropy': CrossEntropyLoss,
    'FocalLoss': FocalLoss,
    'GradientHarmonized': GradientHarmonizedLoss
}

def fetch_loss(name):
    assert name in loss_names
    return loss_names[name]()
