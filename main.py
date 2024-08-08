import numpy as np
import matplotlib.pyplot as plt
import math

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

  C = {}

  while len(S) > 0:
    s = S.pop()
    C.add(s)

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
  C = {s}

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

    C.add(max_s)
    S.remove(max_s)

  return C


def main():
  pass


if __name__ == '__main__':
  main()
