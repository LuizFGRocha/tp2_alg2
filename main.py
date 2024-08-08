import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# S: conjunto de pontos
# C: resultado
# dist(Si, Sj): distância entre Si e Sj
# Queremos minimizar o raio dos clusters: min max dist(Si, Cj)

# todo Pode usar a implementaçãod de norma do numpy?
def dist(a, b, p=2):
  return np.linalg.norm(a - b, ord=p)


def k_clusters_r(S, k, max_r):
  if k >= len(S):
    return S

  C = []

  while len(S) > 0:
    s = S.pop()
    C.append(s)

    for Si in S:
      if dist(Si, s) < 2 * max_r:
        S.remove(Si)

  if len(C) > k:
    return None
  
  return C


def k_clusters(S, k):
  if k >= len(S):
    return S
  
  s = S.pop()
  C = [s]

  while len(S) < k:
    max_dist = 0
    max_s = None

    # Queremos o s que está mais distante de todos os clusters
    for Si in S:
      min_dist = np.inf

      # Acha o cluster mais próximo de Si
      for Cj in C:
        d = dist(Si, Cj)
        if d < min_dist:
          min_dist = d

      # Se a distância é maior que a máxima distância encontrada até agora, atualiza
      if min_dist > max_dist:
        max_dist = min_dist
        max_s = Si

    C.append(max_s)
    S.remove(max_s)

  return C


def cluster_map(S, C):
  map = []

  for Si in S:
    min_dist = np.inf
    min_s = None

    for Cj in C:
      d = dist(Si, Cj)
      if d < min_dist:
        min_dist = d
        min_s = Cj

    map.append(min_s)

  return map


def main():
  #--- Cria os datasets do SKlearn ---#
  n_samples = 500
  seed = 30
  noisy_circles = datasets.make_circles(
      n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
  )
  noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
  blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
  rng = np.random.RandomState(seed)
  no_structure = rng.rand(n_samples, 2), None

  # Anisotropicly distributed data
  random_state = 170
  X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
  transformation = [[0.6, -0.6], [-0.4, 0.8]]
  X_aniso = np.dot(X, transformation)
  aniso = (X_aniso, y)

  # blobs with varied variances
  varied = datasets.make_blobs(
      n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
  )

  # Faz o plot dos datasets

  fig, ax = plt.subplots(1, 6)

  # Lista de datasets
  ds = [noisy_circles, noisy_moons, blobs, aniso, varied, no_structure]

  clusterings = []
  clustering_maps = []

  for i, dataset in enumerate(ds):
    # Para cada dataset, faz os clusterings e printa
    pass

  plt.show()


if __name__ == '__main__':
  main()
