import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from toydatasets import ToyDatasets
from util import read_custom_synthetic

def main():
    n_samples = 500
    seed = 30
    toy = ToyDatasets("./results/toy.csv", "./img")

    toy.add_dataset(*datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
                    2, "moons")

    toy.add_dataset(*datasets.make_blobs(n_samples=n_samples, random_state=seed), 
                    3, "blobs")

    toy.add_dataset(*datasets.make_circles(n_samples=500, random_state=seed), 2, "circles")

    p, l, k = read_custom_synthetic("./syn-instances/test.dat")
    toy.add_dataset(p,l,k,"custom")

    #TODO achar mais datasets

    toy.test_datasets()

if __name__ == '__main__':
    main()