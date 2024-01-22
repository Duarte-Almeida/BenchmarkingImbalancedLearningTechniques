import numpy as np

def random_sample(input_file, output_file):
    data = np.load(input_file)

    np.save(input_file[-4]+"_orig.npy", data)

    num_samples = int(0.1 * len(data))

    selected_indices = np.random.choice(len(data), num_samples, replace=False)

    selected_data = data[selected_indices]

    np.save(output_file, selected_data)

if __name__ == "__main__":
    np.random.seed(42)

    for dataset in ["baf", "mlg"]:
        input_file1 = f'datasets/{dataset}/features_train.npy'
        output_file1 = f'datasets/{dataset}/features_train.npy'

        input_file2 = f'datasets/{dataset}/targets_train.npy'
        output_file2 = f'datasets/{dataset}/targets_train.npy'

        random_sample(input_file1, output_file1)
        print(f"Randomly selected 10% of samples from {input_file1} and saved to {output_file1}")

        random_sample(input_file2, output_file2)
        print(f"Randomly selected 10% of samples from {input_file2} and saved to {output_file2}")
