import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import oversamplers
import datasets
import utils
from sklearn.utils import check_random_state
from pomegranate import BayesianNetwork
import pickle as pkl
from mdlp import MDLP
import os
import sys
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from pgmpy.estimators import TreeSearch
from scipy.stats import entropy
import FaissKNN


def load_data(dataset_name):
    features_train = np.load(f"datasets/{dataset_name}/features_train.npy")
    features_test = np.load(f"datasets/{dataset_name}/features_test.npy")
    targets_train = np.load(f"datasets/{dataset_name}/targets_train.npy")
    targets_test = np.load(f"datasets/{dataset_name}/targets_test.npy")  # Assuming there's a file for test targets
    with open(f"datasets/{dataset_name}/info.pkl", "rb") as fp:
        cat_feats = pkl.load(fp)["cat_feats"]
    return features_train, targets_train, features_test, targets_test, cat_feats

def load_model_configuration(dataset_name, oversampling_method):
    filepath = f"config/{dataset_name}/{oversampling_method}.pkl"
    print(f"Loading from {filepath} to {oversampling_method}")
    with open(filepath, "rb") as fp:
        config = pkl.load(fp)
        print(f"Here is the config: {config}")
    return config

def load_results(dataset_name, oversampling_method):
  
    filepath = f"results/{dataset_name}/preprocessing/{oversampling_method}/{oversampling_method}_scores.npz"
    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(filepath)
    np.load = np_load_old
    return data['train_tpr'], data['test_tpr'], data['train_auc'], data['test_auc']

def compute_probabilities(X, probs, priors, structure):
    #print(f"Computing probabilities...")
    X_nr = X[:, 1:]
    X_r = X[:, [0]]
    X_r_val_dep = X_r.astype(int)
    depend = np.array(structure)
    depend_aux = depend[depend != -1]
    X_nr_dep = X[:, depend_aux]
    X_nr_val_dep = np.dstack((X_nr, X_nr_dep)).astype(int)

    init_probs_nr = np.zeros_like(X_nr).astype(float)
    features_nr = np.array([i for i in range(X.shape[1]) if i != 0])

    for j in range(X_nr.shape[1]):
        ft = features_nr[j]
        #print(f"Feature {ft}")
        #print(f"With parent {depend[ft]}")
        #print(np.unique(X[:, ft]))
        rows = X_nr_val_dep[:, j, 1]
        cols = X_nr_val_dep[:, j, 0]
        init_probs_nr[:, j] = probs[ft][X_nr_val_dep[:, j, 1], X_nr_val_dep[:, j, 0]]

    init_probs_r = probs[0][X_r].reshape(-1)
    init_probs = np.prod(init_probs_nr, axis = 1) * init_probs_r

    return init_probs

def calculate_kl_divergence(X_test, probs_sub, priors_sub, structure_sub, probs_o, priors_o, structure_o):

    sub_probs = compute_probabilities(X_test, probs_sub, priors_sub, structure_sub)
    o_probs = compute_probabilities(X_test, probs_o, priors_o, structure_o)

    #print(f"Probs: {sub_probs}")
    #print(f"Other Probs: {o_probs}")

    return np.mean(np.log(sub_probs / o_probs))

def plot_metric(ax, oversampling_method, train_values, test_values, metric_name):
    # Compute mean values
    mean_train = np.mean(train_values)
    mean_test = np.mean(test_values)

    # Plot train values as a blue cross
    ax.plot([oversampling_method], [mean_train], marker='x', markersize=10, label='Train', color='blue')

    # Plot test values as a red circle
    ax.plot([oversampling_method], [mean_test], marker='o', markersize=10, label='Test', color='red')

    # Connect train and test values with a dashed line
    ax.plot([oversampling_method, oversampling_method], [mean_train, mean_test], linestyle='--', color='gray')

    # Set labels and title
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} - Train/Test Comparison")
    ax.legend()

def subsample_data(X_train, y_train, proportion=0.1, random_state=None):
    
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    pos_subsampled_indices = np.random.choice(pos_indices, size=int(proportion * len(pos_indices)), replace=False)
    neg_subsampled_indices = np.random.choice(neg_indices, size=int(proportion * len(neg_indices)), replace=False)

    indices_subsampled = np.concatenate((pos_subsampled_indices, neg_subsampled_indices))
    np.random.shuffle(indices_subsampled)

    return X_train[indices_subsampled], y_train[indices_subsampled]

def estimate_model(X, X_all):
    if X.shape[0] > 1000:
        idx = np.random.choice(X.shape[0], size=1000, replace=False)
        X_aux = X[idx]
    else:
        X_aux = X
    eps = 1e-5
    all_dicts = np.empty(X.shape[1], dtype='object')
    for j in range(X.shape[1]):
        keys = np.arange(np.max(X_all[:, j] + 1))
        n = keys.shape[0]
        fi = np.zeros(n) + eps
        all_dicts[j] = dict(zip(keys, fi))
    N = X.shape[0]
    priors = []
    #print(f"estimating priors")
    for j in range(X_aux.shape[1]):
        X_class = X_aux[:, j]
        counts = [X_class[X_class == val].shape[0] for val in sorted(all_dicts[j].keys())]
        probs = np.array(counts) / np.sum(np.array(counts))
        probs[probs == 0] = eps
        priors.append(probs)
    #print(f"finishing priors")
    #print(f"estimating structure")
    bayes = BayesianNetwork.from_samples(X_aux, algorithm='chow-liu', root=0)

    #print(f"finished structure")
    depend = []
    for i in bayes.structure:
        if i:
            depend.append(i[0])
        else:
            depend.append(-1)
    
    #print("estimating cpds")
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
                numi = np.where((X_aux[:, j] == l) & (X_aux[:, num_parent] == k))[0].shape[0]
                den = np.where(X_aux[:, num_parent] == k)[0].shape[0]
                if den == 0 or numi == 0:
                    fp[k, l] = eps
                else:
                    fp[k, l] = numi / den

        probs.append(fp)
    #print("finished cpds")

    return probs, priors, depend

def main():

    np.random.seed(42)
    # Argument parser
    parser = argparse.ArgumentParser(description="Analyze oversampling results.")
    parser.add_argument("-dataset", choices=["Base", "baf", "ieee", "mlg"], required=True,
                        help="Dataset name.")
    args = parser.parse_args()

    # Fetch dataset
    X_train, y_train, X_test, y_test, cat_feats = load_data(args.dataset)

    # Oversampling methods
    oversampling_methods = ["Base", "SMOTE", "ADASYN", "KMeansSMOTE", "RACOG"]

    
    # Dict to store KL divergence results
    kl_divergence_results = {method: [] for method in oversampling_methods}  # Exclude "Base"

    probs_histograms = {method: [] for method in oversampling_methods}

    # Create oversampler instances and set parameters
    oversamplers_instances = {}
    for oversampling_method in oversampling_methods:
        if oversampling_method == "Base":
            oversamplers_instances[oversampling_method] = None
            continue
        oversampler = oversamplers.fetch_oversampler(oversampling_method)
        config = load_model_configuration(args.dataset, oversampling_method)
        config = {key[len("oversampler__"): ]: value for key, value in config.items()}
        oversampler.set_params(**config)
        oversampler.set_params(**{"categorical_features": len(cat_feats)})
        oversamplers_instances[oversampling_method] = oversampler

    i_categorical = np.array([i for i in range(X_train.shape[1] - len(cat_feats), X_train.shape[1])])
    # Perform iterations
    for _ in range(n_iter):
        X_sub_o, y_sub_o = subsample_data(X_train, y_train, 0.1, 42)
        X_sub_p, y_sub_p = X_sub_o[y_sub_o == 1], y_sub_o[y_sub_o == 1]
        disc = MDLP(categorical_features=i_categorical, random_state = 42)
        sys.stdout = open(os.devnull, 'w')
        disc.fit(X_sub_p, y_sub_p)
        sys.stdout = sys.__stdout__
        X_sub = disc.transform(X_sub_p, y_sub_p).astype(int)

        probs_sub, priors_sub, structure_sub = estimate_model(X_sub, X_train)
        sub_probs = compute_probabilities(X_sub, probs_sub, priors_sub, structure_sub)
        probs_histograms["Base"].append(sub_probs)

        #sys.exit()

        X_test, y_test = subsample_data(X_train, y_train, 0.1, 42)
        X_test_p, y_test_p = X_test[y_test == 1], y_test[y_test == 1]
        X_test = disc.transform(X_test_p, y_test_p).astype(int)


        for oversampling_method, oversampler in oversamplers_instances.items():
            print(f"Oversampling {oversampling_method}")
            # Apply oversampler to subsampled dataset
            if oversampling_method == "Base":
                X_other, y_other = subsample_data(X_train, y_train, 0.1, 42)
                X_o = X_other[y_other == 1]  
                y_o = y_other[y_other == 1]
                X_o = disc.transform(X_o, y_o).astype(int)
            else:
                X_r, y_r = oversampler.fit_resample(X_sub_o, y_sub_o)
                X_o = X_r[X_sub_o.shape[0]:]
                y_r = y_r[X_sub_o.shape[0]:]
                X_o = disc.transform(X_o, y_o).astype(int)

            probs_o, priors_o, structure_o = estimate_model(X_o, X_train)
            if oversampling_method != "Base":
                o_probs = compute_probabilities(X_o, probs_sub, priors_sub, structure_sub)
                probs_histograms[oversampling_method].append(o_probs)

            # Calculate KL divergence and store results
            kl_divergence = calculate_kl_divergence(X_test, probs_sub, priors_sub, structure_sub, probs_o, priors_o, structure_o)
            kl_divergence_results[oversampling_method].append(kl_divergence)
            print(f"KL divergence: {kl_divergence}")
            #sys.exit()


    # Get the number of oversampling methods
    num_methods = len(oversampling_methods[1:])


    for i, method in enumerate(oversampling_methods[1:]):
        print(f"{i}")
        #print(f"Exploring {method}")
        base_probs = probs_histograms["Base"]
        method_probs = probs_histograms[method]
        #print(f"And probabilities {method_probs}")
        #minimum = np.min([np.min(hist) for hist in base_probs] + [np.min(hist) for hist in method_probs])
        maximum = np.max([np.max(hist) for hist in base_probs] + [np.max(hist) for hist in method_probs])

        n_bins = 50
        base_histogram = np.zeros(n_bins)
        method_histogram = np.zeros(n_bins)

        for i in range(n_iter):
            curr_base = base_probs[i]
            curr_method = method_probs[i]
            base_idx = np.floor(curr_base / (maximum / n_bins)).astype(int)
            base_idx[base_idx == n_bins] = n_bins - 1
            method_idx = np.floor(curr_method / (maximum / n_bins)).astype(int)
            method_idx[method_idx == n_bins] = n_bins - 1
            base_counts = np.bincount(base_idx)
            base_counts = np.concatenate((base_counts, np.zeros(n_bins - base_counts.shape[0])))
            method_counts = np.bincount(method_idx)
            method_counts = np.concatenate((method_counts, np.zeros(n_bins - method_counts.shape[0])))
            base_histogram += (base_counts / np.sum(base_counts))
            method_histogram += (method_counts / np.sum(method_counts))
        
        method_histogram /= (n_iter)
        base_histogram /= (n_iter)

        # Define the x values
        x_values_base = np.linspace(0, maximum, n_bins)
        x_values_method = np.linspace((maximum / (n_bins)), maximum + (maximum / (n_bins)), n_bins)

        # Plotting the histograms with alpha=0.5 and log scale on y-axis
        plt.bar(x_values_base, base_histogram, alpha=0.5, width = maximum / (2 * n_bins), label='Base', align = "edge")
        plt.bar(x_values_method, method_histogram, alpha=0.5, width = -maximum / (2 * n_bins), label=f'{method}', align = "edge")

        # Add title to each subplot
        plt.title(method)

        # Set axis labels
        plt.xlabel('Density')
        plt.ylabel('Probability')

        plt.savefig(f"analysis/{method}_densities.pdf")
        plt.clf()

    
    # Visualize KL divergences with box plots
    plt.figure(figsize=(10, 6))
    plt.boxplot([kl_divergence_results[method] for method in kl_divergence_results.keys()],
                labels=list(kl_divergence_results.keys()))
    #plt.title('KL Divergence for Different Oversampling Methods')
    plt.xlabel('Oversampling Methods')
    plt.ylabel('KL Divergence')
    plt.savefig(f"analysis/KL_oversamplers.pdf")
    plt.clf()

    plt.figure(figsize=(6.4, 4.8))
    

    n_iter = 10

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    knn = FaissKNN.FaissKNN(n_neighbors = 10 + 1)

    nn_histograms = {method: np.zeros(10 + 1) for method in oversampling_methods}

    for _ in range(n_iter):
        X_sub_o, y_sub_o = subsample_data(X_train, y_train, 0.1, 42)
        n_idx = np.where(y_sub_o == 0)[0]

        X_cat = encoder.fit_transform(X_sub_o[:, -len(cat_feats):])
        X_non_cat = X_sub_o[:, :len(cat_feats)]
        std_devs = np.std(X_non_cat, axis = 1)
        X_cat = X_cat * (np.median(std_devs) / np.sqrt(2))
    
        X_sub_encoded = np.concatenate((X_non_cat, X_cat), axis=1)
        X_sub_n = X_sub_encoded[n_idx]
        knn.fit(X_sub_encoded)

        # Find distances and indices of k nearest neighbors for all examples
        distances, indices = knn.kneighbors(X_sub_n)

        # Exclude the first index (itself) and count positive neighbors
        counts_sub = np.bincount(np.sum(y_sub_o[indices[:, 1:]] == 1, axis=1))
        counts_sub = np.concatenate((counts_sub, np.zeros(11 - counts_sub.shape[0])))
        print(counts_sub)
        counts_sub /= np.sum(counts_sub)

        # Extract counts for negative examples
        print(f"Here is the result for Base: {counts_sub}")
        nn_histograms["Base"] += counts_sub
        #sys.exit()

        for oversampling_method, oversampler in oversamplers_instances.items():
            print(f"Oversampling {oversampling_method}")
            # Apply oversampler to subsampled dataset
            if oversampling_method == "Base":
                continue
            else:
                X_r, y_r = oversampler.fit_resample(X_sub_o, y_sub_o)
                X_cat = encoder.fit_transform(X_r[:, -len(cat_feats):])
                X_non_cat = X_r[:, :len(cat_feats)]
                std_devs = np.std(X_non_cat, axis = 1)
                X_cat = X_cat * (np.median(std_devs) / np.sqrt(2))
                X_r = np.concatenate((X_non_cat, X_cat), axis=1)
                X_r_n = X_r[y_r == 0]
                knn.fit(X_r)
                distances, indices = knn.kneighbors(X_r_n)
                counts_sub = np.bincount(np.sum(y_r[indices[:, 1:]] == 1, axis=1))
                counts_sub = np.concatenate((counts_sub, np.zeros(11 - counts_sub.shape[0])))
                counts_sub /= np.sum(counts_sub)

                # Extract counts for negative examples
                print(f"Here is the result for {oversampling_method}: {counts_sub}")
                nn_histograms[oversampling_method] += counts_sub


    for method in oversampling_methods[1:]:
        base_probs = nn_histograms["Base"]
        method_probs = nn_histograms[method]
        maximum = 10
        n_bins = 10

        # Define the x values
        x_values_base = np.linspace(0, maximum, n_bins + 1)
        x_values_method = np.linspace(0, maximum, n_bins + 1)

        print(f"{x_values_base}")
        print(f"{x_values_method}")

        # Plotting the histograms with alpha=0.5
        plt.bar(x_values_base, base_probs, alpha=0.5, width = -maximum / (2 * n_bins * 1.5), label='Base', align = "edge")
        plt.bar(x_values_method, method_probs, alpha=0.5, width = maximum / (2 * n_bins * 1.5), label=f'{method}', align = "edge")

        # Add legend
        plt.legend()

        # Set the ticks on the x-axis
        plt.xticks(np.arange(0, maximum + 1, 1))
        plt.yscale('log')

        # Add title to each subplot
        plt.title(method)

        # Set axis labels
        plt.xlabel('Density')
        plt.ylabel('# of positive NN')

        # Show the plot
        plt.savefig(f"analysis/{method}_KNN.pdf")
        plt.clf()
    
if __name__ == "__main__":
    os.makedirs("analysis/", exist_ok=True)
    main()
