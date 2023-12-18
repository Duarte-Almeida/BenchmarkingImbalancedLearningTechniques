from .SMOTE_IPF import SMOTEIPFWrapper

hybrid_names = {
    'IPF':SMOTEIPFWrapper 
}

def fetch_hybrid(name):
    assert name in hybrid_names
    return hybrid_names[name]()