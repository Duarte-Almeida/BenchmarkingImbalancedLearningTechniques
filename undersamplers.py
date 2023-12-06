from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler

class RUSWrapper(RandomUnderSampler):

    def parameter_grid(self):
        return {
            'sampling_strategy': [1.0]
        }


undersampler_names = {
    'RUS': RUSWrapper
}

def fetch_undersampler(name):
    assert name in undersampler_names
    return undersampler_names[name]()
