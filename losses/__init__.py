from .CrossEntropyLoss import CrossEntropyLoss
from .FocalLoss import FocalLoss
from .GradientHarmonizedLoss import GradientHarmonizedLoss

loss_names = {
    'WeightedCrossEntropy': CrossEntropyLoss,
    'LabelSmoothing': CrossEntropyLoss,
    'LabelRelaxation': CrossEntropyLoss,
    'FocalLoss': FocalLoss,
    'GradientHarmonized': GradientHarmonizedLoss
}

actions = {
    'WeightedCrossEntropy':  lambda clf: clf.untoggle_param_grid("weighted"),
    'LabelSmoothing':  lambda clf: clf.untoggle_param_grid("ls"),
    'LabelRelaxation': lambda clf: clf.untoggle_param_grid("lr"),
    'FocalLoss': None,
    'GradientHarmonized': None
}



def fetch_loss(name):
    assert name in loss_names
    res = loss_names[name]()
    action = actions[name]
    if action is not None:
        action(res)
    return res
