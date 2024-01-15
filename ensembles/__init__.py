from .BalanceBaggingWrapper import BalanceBaggingWrapper
from .EasyEnsembleWrapper import EasyEnsembleWrapper
from .MesaWrapper import MesaWrapper
from .SelfPacedEnsembleWrapper import SelfPacedEnsembleWrapper
from .StackedEnsemble import StackedEnsemble

ensemble_names = {
    #'BalanceBagging':BalanceBaggingWrapper,
    #'EasyEnsemble':EasyEnsembleWrapper,
    #'MESA':MesaWrapper,
    'SelfPaced':SelfPacedEnsembleWrapper,
    'StackedEnsemble':StackedEnsemble,
}

def fetch_ensemble(name):
    assert name in ensemble_names
    return ensemble_names[name]()