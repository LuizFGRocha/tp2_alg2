import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from toydatasets import ToyDatasets

def main():
    n_samples = 500
    seed = 30
    toy = ToyDatasets("./results/toy.csv", "./img")

    points, labels = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    toy.add_dataset(points, labels, 2, "moons")
    toy.test_datasets()


if __name__ == '__main__':
    main()