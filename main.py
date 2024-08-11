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

def gen_scikit_datasets(toy, n_samples = 500, seed = 30):
    'Adds scikit datasets to toy'
    toy.add_dataset(*datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
                    2, "moons")

    toy.add_dataset(*datasets.make_blobs(n_samples=n_samples, random_state=seed), 
                    3, "blobs")

    toy.add_dataset(*datasets.make_circles(n_samples=500, noise=0.05, random_state=seed), 2, "circles")

    #Clusters de dados em dimensao maior -> Parecido com o blobs, mas em alta dimensao

    #Varios valores informativos
    toy.add_dataset(*datasets.make_classification(n_samples, n_redundant=0, n_informative=10, n_features=10, random_state=seed), 2, "high-order-informative")

    #Alguns valores redundantes
    toy.add_dataset(*datasets.make_classification(n_samples, n_redundant=3, n_informative=7, n_features=10,random_state=seed), 2, "ho-slightly-redundant")

    #Grande parte dos valores redundantes
    toy.add_dataset(*datasets.make_classification(n_samples, n_redundant=7, n_informative=3, n_features=10,random_state=seed), 2, "ho-redundant")

    #Grande parte dos valores sendo inuteis
    toy.add_dataset(*datasets.make_classification(n_samples, n_redundant=0, n_informative=4, n_features=10,random_state=seed), 2, "ho-useless")

    #Caso geral - numero razoavel de inuteis, repetidos, redundantes e informativos
    toy.add_dataset(*datasets.make_classification(n_samples, n_redundant=2, n_repeated=2 ,n_informative=5, n_features=10,random_state=seed), 2, "ho-general")

    #Anisotropic
    p, l = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    p = np.dot(p, transformation)
    toy.add_dataset(p,l,3,"aniso")

    rng = np.random.RandomState(seed)
    toy.add_dataset(rng.rand(n_samples,2), rng.randint(2, size=n_samples), 2, "random")

def main():
    toy = ToyDatasets("./results/toy.csv", "./img")

    #Criar todas as sinteticas do scikit
    gen_scikit_datasets(toy)

    #Ler todas as sinteticas custom
    read_all_synthetic(toy, "./syn-instances")

    toy.test_datasets()

if __name__ == '__main__':
    main()