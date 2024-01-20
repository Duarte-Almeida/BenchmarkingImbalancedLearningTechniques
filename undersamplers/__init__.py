from .RUS import RUSWrapper
from .NCR import NCRWrapper
from .InstanceHardness import IHTWrapper

undersampler_names = {
    'RUS': RUSWrapper,
    'NCR': NCRWrapper,
    'IHT': IHTWrapper,
}

def fetch_undersampler(name):
    assert name in undersampler_names
    return undersampler_names[name]()