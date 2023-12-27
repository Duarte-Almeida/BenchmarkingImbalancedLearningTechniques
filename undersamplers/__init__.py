from .TomekLinks import TomekLinksWrapper
from .ENN import ENNWrapper
from .RUS import RUSWrapper
from .NCR import NCRWrapper
from .InstanceHardness import IRWrapper

undersampler_names = {
    'RUS': RUSWrapper,
    'NCR': NCRWrapper,
    'InstanceHardness': IRWrapper
}

def fetch_undersampler(name):
    assert name in undersampler_names
    return undersampler_names[name]()