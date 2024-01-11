from .BalanceBaggingWrapper import BalanceBaggingWrapper
from .EasyEnsembleWrapper import EasyEnsembleWrapper
from .MesaWrapper import MesaWrapper
from .SelfPacedEnsembleWrapper import SelfPacedEnsembleWrapper

ensemble_names = {
    'BB':BalanceBaggingWrapper,
    'EE':EasyEnsembleWrapper,
    'Mesa':MesaWrapper,
    'SP':SelfPacedEnsembleWrapper
}

def fetch_ensemble(name, estimator, n_estimators=10, random_state=42, **kwargs):
    assert name in ensemble_names
    return ensemble_names[name](estimator=estimator, n_estimators=n_estimators, random_state=random_state, **kwargs)