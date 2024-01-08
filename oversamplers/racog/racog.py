'''
   Code adapted from: https://github.com/airysen/racog
'''

import numpy as np
import pandas as pd

from pomegranate import BayesianNetwork

from imblearn.over_sampling.base import BaseOverSampler
from caimcaim import CAIMD
from mdlp import MDLP
from tqdm import *

from functools import partial
import multiprocessing

import sys, os


MAX_DATASET_SIZE = 10000

class RACOG(BaseOverSampler):
    """
    RACOG oversampling class

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    discretization: 'caim' or 'mdlp'
        Method for discretization continuous variables

    categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical (not using discretization).
        - 'auto' (default): Only those features whose number of unique values exceeds
                            the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe

    lag0: int
        Lag is the number of consecutive samples that are discarded from
        the Markov chain following each accepted sample to avoid
        autocorrelation between consecutive samples

    offset: int
        Warm up for gibbs sampler. It is number of sample generation iterations
        that are needed for the samples to reach a stationary distribution
        
    root: int
        Index of the root feature using in Chow-Liu algorithm

    only_sampled: bool
        Concatenate or not original data with new samples. If True,
        only new samples will return

    threshold: int
        If number of samples is needed for oversampling less than threshold,
        no oversamping will be made for that class

    eps: float
        A very small value to replace zero values of any probability

    verbose: int
        If greather than 0, enable verbose output

    shuffle: bool
        Shuffle or not the original data and a sampled array. If 'False',
        new rows will be stacked after original rows

    n_jobs: int
        The number of jobs to run in parallel for samplng

    References
    ----------
    [1] B. Das, N. C. Krishnan and D. J. Cook,
        "RACOG and wRACOG: Two Probabilistic Oversampling Techniques,"
        in IEEE Transactions on Knowledge and Data Engineering,
        vol. 27, no. 1, pp. 222-234, Jan. 1 2015.
        doi: 10.1109/TKDE.2014.2324567
        http://ieeexplore.ieee.org/document/6816044/

    [2] João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
        "An embedded imputation method via Attribute-based Decision Graphs",
        Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
        ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2016.03.027.
        http://www.sciencedirect.com/science/article/pii/S0957417416301208

    Example
    ---------

    """

    def __init__(self, sampling_strategy='auto', random_state=42, discretization='mdlp', categorical_features='auto',
                 offset=50, lag=20, root=0, only_sampled = False,
                 threshold=10, eps=10E-5, verbose=2, shuffle=False, n_jobs=1):
        
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.discretization = discretization
        self.categorical_features = categorical_features
        if not categorical_features:
            self.categorical_features = 'auto'
        self.i_categorical = []  # store only index of columns with categorical features
        self.only_sampled = only_sampled

        self.root = root

        self.offset = offset
        self.lag = lag
        self.threshold = threshold
        self.eps = eps

        self.verbose = verbose
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

    def fit(self, X, y):
        self.i_categorical = []
        self.pdflag = False
        if isinstance(X, pd.DataFrame):
            if isinstance(self.categorical_features, list):
                self.i_categorical = [X.columns.get_loc(label) for label in self.categorical]
            X = X.values
            y = y.values
            self.pdflag = True
        X_di = X
        super().fit(X, y)

        # get indices of categorical features
        if self.categorical_features != 'all':
            if self.categorical_features == 'auto':
                self.i_categorical = self.check_categorical(X, y)
            elif (isinstance(self.categorical_features, list)) or (isinstance(self.categorical_features, np.ndarray)):
                if not self.pdflag:
                    self.i_categorical = self.categorical_features[:]
            elif isinstance(self.categorical_features, int):
                self.i_categorical = np.arange(X.shape[1] - self.categorical_features, X.shape[1])
        else:
            self.i_categorical = np.arange(X.shape[1]).tolist()

        # perform discretization
        continuous = self.i_categorical
        if continuous.size != 0:
            if self.discretization == 'mdlp':
                self.disc = MDLP(categorical_features=self.i_categorical, random_state = self.random_state)

            # consider subsampled dataset in discretization process to speedup
            if (X.shape[0] > MAX_DATASET_SIZE):
                np.random.seed(self.random_state)
                idx = np.random.choice(X.shape[0], size = MAX_DATASET_SIZE, replace = False)
                X_sub = X_di[idx]
                y_sub = y[idx]
            sys.stdout = open(os.devnull, 'w')
            self.disc.fit(X_sub, y_sub)
            sys.stdout = sys.__stdout__
            X_di = self.disc.transform(X_di, y)
            X_di = X_di.astype(int)

        # initialize tree structure, prior and conditional probability tables
        self.probs = {}
        self.priors = {}
        self.structure = {}
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples < self.threshold:
                continue
            probs, priors, depend = self._tan(X_di, y, X_di[y == class_sample])
            self.probs[class_sample] = probs
            self.priors[class_sample] = priors
            self.structure[class_sample] = depend
        
        return self

    def sample(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.pdflag = True
            self.pd_index = X.index.values
            self.pd_columns = X.columns
            self.pd_yname = y.name
            X = X.values
            y = y.values
        return self._sample(X, y)

    def _sample(self, X, y):
        n, m = X.shape
        dtype = X.dtype
        offset = self.offset
        n_jobs = self.n_jobs
        lag0 = self.lag
        X_resampled = np.zeros((0, m), dtype=dtype)
        y_resampled = np.array([])

        X_di = X

        continuous = self.i_categorical
        if continuous.size != 0:
            X_di = self.disc.transform(X)

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples < self.threshold:
                continue

            X_class = X_di[y == class_sample]
            priors = self.priors[class_sample]
            depend = self.structure[class_sample]
            probs = self.probs[class_sample]        

            n_iter = int(np.ceil(n_samples / X_class.shape[0])) * lag0 + offset
            val_list = self._get_vlist(priors)

            X_new = self._multi_run(X_class=X_class, vlist=val_list,
                                    depend=depend, probs=probs,
                                    priors=priors, n_iter=n_iter)

            continuous = self.i_categorical
            if continuous.size != 0:
                X_new = self._recon_continuous(X, y, X_di, X_new, class_sample, self.i_categorical)
            y_new = np.ones(X_new.shape[0]) * class_sample
            X_resampled = np.vstack((X_resampled,
                                     X_new[:n_samples]))

            y_resampled = np.hstack((y_resampled, y_new[:n_samples])) if y_resampled.size else y_new[:n_samples]

        if self.only_sampled is False:
            if self.pdflag:
                index = self.pd_index[-1] + np.arange(X_resampled.shape[0])
                X_resampled, y_resampled = self.create_pandas(X_resampled, y_resampled, index=index)
                X, y = self.create_pandas(X, y)

                X_resampled = pd.concat((X, X_resampled), axis=0)
                y_resampled = pd.concat((y, y_resampled))
            else:
                X_resampled = np.vstack((X,
                                         X_resampled))

                y_resampled = np.hstack((y,
                                         y_resampled))
        else:
            if self.pdflag:
                index = self.pd_index[-1] + np.arange(X_resampled.shape[0])
                X_resampled, y_resampled = self.create_pandas(X_resampled, y_resampled, index=index)
            else:
                pass

        if self.shuffle is True:
            X_resampled, y_resampled = self.shuffle_rows(X_resampled, y_resampled, num=5)

        return X_resampled, y_resampled

    def _recon_continuous(self, X, y, X_disc, X_sampled, class_sample, categorical):
        """
        Construct continuous features from categorical ones
        """     
        print(f"Performing reconstruction of continuous variables")
        X_recon = np.zeros(X_sampled.shape)
        cut_points = self.disc.cut_points_

        # perform continuous reconstruction assuming uniform distribution within each bin
        for j in tqdm(range(X_sampled.shape[1])):
            if j in categorical:
                X_recon[:, j] = X_sampled[:, j]
                continue
            X_sampled_j = X_sampled[:, j]
            min_j = np.min(X[:, j])
            max_j = np.max(X[:, j]) 
            
            # upper limits of each bin (in the last bin it is the maximum of that feature)
            upper = np.zeros(X_sampled_j.shape[0])
            non_edge_indices = (X_sampled_j != len(cut_points[j]))
            edge_indices = (X_sampled_j == len(cut_points[j]))
            upper[edge_indices] = max_j
            upper[non_edge_indices] = cut_points[j][X_sampled_j[non_edge_indices]]
            
            # lower limits of each bin (in the first it is the minimum of that feature)
            lower = np.zeros(X_sampled_j.shape[0])
            non_edge_indices = (X_sampled_j != 0)
            edge_indices = (X_sampled_j == 0)
            lower[edge_indices] = min_j
            lower[non_edge_indices] = cut_points[j][X_sampled_j[non_edge_indices] - 1]

            np.random.seed(self.random_state)
            X_recon[:, j] = lower + np.random.uniform() * (upper - lower)

        return X_recon

    def _multi_run(self, X_class, vlist, depend, probs, priors, n_iter):
        """
        Run Gibbs samplers in parallel
        """
        m = X_class.shape[1]
        n_jobs = self.n_jobs

        params = {'vlist': vlist,
                  'depend': depend,
                  'probs': probs,
                  'priors': priors,
                  'T': n_iter}
        X_new = np.zeros((0, m))
        p = multiprocessing.Pool(n_jobs)
        zi = X_class

        chunk_size = len(X_class) // n_jobs
        chunks = [X_class[i:i+chunk_size] for i in range(0, len(X_class), chunk_size)]
        
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.starmap(self._gibbs_sampler, [(chunk, vlist, depend, probs, priors, n_iter) for chunk in chunks])
        for s in results:
            X_new = np.vstack((X_new, np.array(s)))
        
        return X_new.astype(int)

    def _all_keys(self, X, eps=0.00001):
        """
        Get all possible values for each feature
        """
        all_dicts = np.empty(X.shape[1], dtype='object')
        for j in range(X.shape[1]):
            keys = np.arange(np.max(X[:, j] + 1))
            n = keys.shape[0]
            fi = np.zeros(n) + eps
            all_dicts[j] = dict(zip(keys, fi))
        return all_dicts

    def _get_structure(self, X_plus, root=0):
        """
        Get the features dependency structure of the minority class
        """
        bayes = BayesianNetwork.from_samples(X_plus, algorithm='chow-liu', root=root)
        depend = []
        for i in bayes.structure:
            if i:
                depend.append(i[0])
            else:
                depend.append(-1)
        return depend

    def _fill_priors(self, X_plus, all_dicts):
        """
        Get priors of the minority class for each feature
        """
        N = X_plus.shape[0]
        priors = []
        for j in range(X_plus.shape[1]):

            X_class = X_plus[:, j]
            counts = [X_class[X_class == val].shape[0] for val in sorted(all_dicts[j].keys())]
            probs = np.array(counts) / np.sum(np.array(counts))
            probs[probs == 0] = self.eps
            priors.append(probs)

        return priors

    def _tan(self, X, y, X_plus):
        """
        Construct prior and dependency tables of the minority class from Chow-Liu dependence tree
        """
        allkeys = self._all_keys(X, eps=self.eps)
        priors = self._fill_priors(X_plus, allkeys)
        depend = self._get_structure(X_plus, root=self.root)

        probs = []  # np.empty(X.shape[1], dtype='object')
        for j in range(len(depend)):

            if depend[j] == -1:
                probs.append(priors[j])
                continue
            num_parent = depend[j]

            lend = len(priors[num_parent])
            leni = len(priors[j])
            fp = np.zeros((lend, leni))
            for k in range(lend):
                for l in range(leni):
                    numi = np.where((X_plus[:, j] == l) & (X_plus[:, num_parent] == k))[0].shape[0]
                    den = np.where(X_plus[:, num_parent] == k)[0].shape[0]
                    if den == 0 or numi == 0:
                        fp[k, l] = self.eps
                    else:
                        fp[k, l] = numi / den

            probs.append(fp)
    
        return probs, priors, depend

    def _gibbs_sampler(self, zi, vlist, depend, probs, priors, T):
        """
        Gibbs sampler
        """
        lag0 = self.lag
        offset = self.offset
        sample = []

        # Get the root, leaf and intermediate nodes from the bayesian tree structure
        Zi = zi.astype(int)
        sample = []
        depend = np.array(depend)
        root = np.argwhere(depend == -1)[0][0]
        leaves = [i for i in range(len(depend)) if i not in depend]
        descendants = []
        for i in range(len(depend)):
            res = np.where(depend == i)[0]
            if res.size == 0:
                descendants.append(-1)
            else:
                descendants.append(res)

        # Compute initial probabilities
        zi_nr = np.delete(zi, root, axis = 1)
        zi_r = zi[:, [root]]
        depend_aux = depend[depend != -1]
        zi_nr_dep = zi[:, depend_aux]
        zi_nr_val_dep = np.dstack((zi_nr, zi_nr_dep)).astype(int)
        features_nr = np.array([i for i in range(zi.shape[1]) if i != root])

        zi_r_val_dep = zi_r.astype(int)

        init_probs_nr = np.zeros_like(zi_nr)

        for j in range(zi_nr.shape[1]):
            ft = features_nr[j]
            rows = zi_nr_val_dep[:, j, 1]
            cols = zi_nr_val_dep[:, j, 0]
            init_probs_nr[:, j] = probs[ft][zi_nr_val_dep[:, j, 1], zi_nr_val_dep[:, j, 0]]

        init_probs_r = probs[root][zi_r_val_dep].reshape(-1)

        init_probs = np.prod(init_probs_nr, axis = 1) * init_probs_r

        curr_probs = init_probs
        aux = [i for i in range(zi.shape[0])]

        def vectorized_random_choice(prob_matrix):
            s = prob_matrix.cumsum(axis=0)
            np.random.seed(self.random_state)
            r = np.random.rand(prob_matrix.shape[1])
            k = (s < r).sum(axis=0)
            return k - 1
        
        # perform random sampling
        for t in tqdm(range(T)):
            for j in range(zi.shape[1]):
                
                # root node: divide by the old prior, multiply by all possible new priors
                if j == root:
                    old_priors = probs[j][Zi[:, j]]
                    new_priors = probs[j][vlist[j]]
                    curr_probs = curr_probs / old_priors
                    probs_aux = np.outer(curr_probs, new_priors)
                    probs_aux = probs_aux / np.sum(probs_aux, axis = 1)[:, np.newaxis]
                    #print(probs_aux)

                    #probs_aux = np.repeat([new_priors], repeats = Zi.shape[0], axis = 0)
                    if (np.any(np.sum(probs_aux, axis = 1)[:, np.newaxis] == 0)):
                        print("One")
                        print(t)
                        print(np.sum(probs_aux, axis = 1)[:, np.newaxis])
                    
                    idx = vectorized_random_choice(probs_aux.T)
                    curr_probs = curr_probs * probs_aux[aux, idx]
                    zi[:, j] = idx

                # leaf nodes: divide by old cond prob, multiply by all possible new cond probs
                elif j in leaves:

                    dep = depend[j]
                    old_probs = probs[j][Zi[:, dep], Zi[:, j]]

                    idx_row = np.repeat(Zi[:, [dep]], repeats = len(vlist[j]), axis = 1)
                    idx_col = np.repeat([vlist[j]], repeats = Zi.shape[0], axis = 0)

                    new_probs = probs[j][idx_row, idx_col]

                    curr_probs = (curr_probs / old_probs).reshape(-1, 1)
                    probs_aux = curr_probs * new_probs
                    if (np.any(np.sum(probs_aux, axis = 1)[:, np.newaxis] == 0)):
                        print("Two")
                        print(t, j)
                        print(np.sum(probs_aux, axis = 1)[:, np.newaxis])

                    probs_aux = probs_aux / np.sum(probs_aux, axis = 1)[:, np.newaxis]
                    idx = vectorized_random_choice(probs_aux.T)
                    curr_probs = probs_aux[aux, idx]
                    zi[:, j] = idx

                # other nodes: divide by old cond prob, multiply by all possible new cond probs
                else:
                    dep = depend[j]
                    
                    old_probs = probs[j][Zi[:, dep], Zi[:, j]]
                    curr_probs = curr_probs / old_probs

                    for desc in descendants[j]:
                        old_probs = probs[desc][Zi[:, j], Zi[:, desc]]
                        curr_probs = curr_probs / old_probs

                    idx_row = np.repeat(Zi[:, [dep]], repeats = len(vlist[j]), axis = 1)
                    idx_col = np.repeat([vlist[j]], repeats = Zi.shape[0], axis = 0)

                    new_probs = probs[j][idx_row, idx_col]
                    probs_aux = curr_probs.reshape(-1, 1) * new_probs

                    for desc in descendants[j]:
                        idx_row = np.repeat([vlist[j]], repeats = Zi.shape[0], axis = 0)
                        idx_col = np.repeat(Zi[:, [desc]], repeats = len(vlist[j]), axis = 1)

                        new_probs = probs[desc][idx_row, idx_col]
                        probs_aux = probs_aux * new_probs
                    
                    if (np.any(np.sum(probs_aux, axis = 1)[:, np.newaxis] == 0)):
                        print("Three")
                        print(t, j)
                        print(np.sum(probs_aux, axis = 1)[:, np.newaxis])

                    probs_aux = probs_aux / np.sum(probs_aux, axis = 1)[:, np.newaxis]

                    idx = vectorized_random_choice(probs_aux.T)
                    curr_probs = probs_aux[aux, idx]
                    zi[:, j] = idx

            if t > offset and t % lag0 == 0:
                sample.extend(Zi)

        return np.array(sample)

    def shuffle_rows(self, X, y, num=5, random_state=None):
        """
        Shuffle rows of input arrays
        """
        if isinstance(X, pd.DataFrame):
            new_index = X.index.values
            for i in range(num):
                if random_state is not None:
                    np.random.seed(random_state + i)
                new_index = np.random.permutation(new_index)
            X.reindex(new_index)
            y.reindex(new_index)
        else:
            for i in range(num):
                if random_state is not None:
                    np.random.seed(random_state + i)
                new_index = np.random.permutation(new_index)
                X = X[new_index]
                y = y[new_index]
        return X, y

    def _get_vlist(self, priors):
        val_list = []
        eps = self.eps

        for j in range(len(priors)):
            pr = priors[j]
            one = [i for i in range(len(pr))]
            val_list.append(one)
        return val_list

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 2 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical

    def create_pandas(self, X, y, index=None, columns=None, yname=None):
        """
        Create pandas dataframe
        """
        if index is None:
            index = self.pd_index
        if columns is None:
            columns = self.pd_columns
        if yname is None:
            yname = self.pd_yname
        X = pd.DataFrame(X, columns=columns, index=index)
        y = pd.Series(y, index=index, name=yname)
        return X, y


class WrongDistributionException(Exception):
    # Raise if wrong type of Distribution
    pass


class TooManyValues(Exception):
    # Raise if additional binning is needed
    pass
