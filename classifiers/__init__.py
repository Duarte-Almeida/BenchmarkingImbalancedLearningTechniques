from .LGBM import LGBM
from .NeuralNetwork import NeuralNetwork

loss_names = {
    'Base': LGBM,
}

actions = {
    'Base': None,
}



def fetch_clf(name):
    assert name in loss_names
    res = loss_names[name]()
    action = actions[name]
    if action is not None:
        action(res)
    return res
