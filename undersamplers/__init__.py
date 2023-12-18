from .TomekLinks import TomekLinksWrapper
from .ENN import ENNWrapper
from .RUS import RUSWrapper

undersampler_names = {
    'RUS': RUSWrapper,
    'TomekLinks': TomekLinksWrapper,
    'ENN': ENNWrapper
}

def fetch_undersampler(name):
    assert name in undersampler_names
    return undersampler_names[name]()