from .MesaWrapper import MesaWrapper
from .SelfPacedEnsembleWrapper import SelfPacedEnsembleWrapper
from .StackedEnsemble import StackedEnsemble

ensemble_names = {
    'MESA':MesaWrapper,
    'SelfPaced':SelfPacedEnsembleWrapper,
    'StackedEnsemble':StackedEnsemble,
}

def fetch_ensemble(name):
    assert name in ensemble_names
    return ensemble_names[name]()