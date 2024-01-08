from .TomekLinks import TomekLinksWrapper
from .ENN import ENNWrapper
from .RUS import RUSWrapper
from .NCR import NCRWrapper
from .InstanceHardness import IHTWrapper
from .IPF import IPFWrapper

undersampler_names = {
    'RUS': RUSWrapper,
    'NCR': NCRWrapper,
    'IHT': IHTWrapper,
    'ENN': ENNWrapper,
    'TomekLinks': TomekLinksWrapper,
    'IPF': IPFWrapper
}

def fetch_undersampler(name):
    assert name in undersampler_names
    return undersampler_names[name]()