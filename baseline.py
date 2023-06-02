import pandas as pd
import numpy as np

names = {
    "dst": np.int32,
    "hist": np.int32,
    # "enumber": np.int32,
    "etime": np.float64,
    "rnumber": np.int32,
}


def load_data():
    data = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[2, 3, 5, 6],
        names=list(names),
        dtype=names,
    )
    return data


def add_gaussian_noise(data, is_same_noise=False):
    data = data.astype(np.float64)
    noisy_data = data.copy()
    for name in names.keys():
        noise_std = 100 if is_same_noise else 0.05 * data[name].mean()
        noisy_data[name] = data[name] + np.random.normal(0, noise_std, data.shape[0])
    return noisy_data


def compute_mse(data, noisy_data):
    return ((data - noisy_data) ** 2).mean()


def compute_mape(data, noisy_data):
    return (np.abs(data - noisy_data) / data).mean()


def evaluate():
    data = load_data()
    noisy_data = add_gaussian_noise(data)
    mse = compute_mse(data, noisy_data)
    mape = compute_mape(data, noisy_data)
    print(f"MSE:\n {mse}\n")
    print(f"MAPE:\n {mape}")


if __name__ == "__main__":
    evaluate()
