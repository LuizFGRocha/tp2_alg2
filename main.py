import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from toydatasets import ToyDatasets
from util import read_custom_synthetic

def read_all_synthetic(toy, dir):
    'Reads all instances in directory and add them as dataset in toy'
    files = os.listdir(dir)
    
    for file in files:
        p,l,k = read_custom_synthetic(f"{dir}/{file}")
        toy.add_dataset(p,l,k,file.split(".")[0])

def main():
    n_samples = 500
    seed = 30
    toy = ToyDatasets("./results/toy.csv", "./img")

    toy.add_dataset(*datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
                    2, "moons")

    toy.add_dataset(*datasets.make_blobs(n_samples=n_samples, random_state=seed), 
                    3, "blobs")

    toy.add_dataset(*datasets.make_circles(n_samples=500, random_state=seed), 2, "circles")

    p, l = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    p = np.dot(p, transformation)
    toy.add_dataset(p,l,3,"aniso")

    #Ler todas as sinteticas
    read_all_synthetic(toy, "./syn-instances")

    toy.test_datasets()

if __name__ == '__main__':
    main()