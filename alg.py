import numpy as np

# S: conjunto de pontos
# C: resultado, pontos que representam centros
# dist[Si, Sj]: distância entre Si e Sj, armazenada em uma matriz de distancias
# Queremos minimizar o raio dos clusters: min max dist(Si, Cj)

class Instance:
    def __init__(self, S, k, dist_matrix):
        self.S = S
        self.k = k
        self.dist_matrix = dist_matrix

    def k_clusters_r(self, max_r):
        'Solves K-means using max radius 2-approximation algorithm'
        S = np.random.permutation(self.S) # Permute S for different results
        if self.k >= len(S):
            return S

        C = []

        while len(S) > 0:
            s = S.pop()
            C.append(s)

            for Si in S:
                if self.dist[Si, s] < 2 * max_r:
                    S.remove(Si)

        if len(C) > self.k:
            return None
        
        return C

    def k_clusters(self):
        'Solves K-means through greedy 2-approximation algorithm'
        S = np.random.permutation(self.S)
        if self.k >= len(S):
            return S
    
        s = S.pop()
        C = [s]

        while len(C) < self.k:
            max_dist = 0
            max_s = None

            # Queremos o s que está mais distante de todos os clusters
            for Si in S:
                min_dist = np.inf

                # Acha o cluster mais próximo de Si
                for Cj in C:
                    d = self.dist[Si, Cj]
                    if d < min_dist:
                        min_dist = d

                # Se a distância é maior que a máxima distância encontrada até agora, atualiza
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_s = Si

            C.append(max_s)
            S.remove(max_s)

            return C

    def cluster_map(self, C):
        'Maps each element from S to a cluster in C, according to its lowest distance.'
        map = []

        for Si in self.S:
            min_dist = np.inf
            min_s = None

            for Cj in C:
                d = self.dist[Si, Cj]
                if d < min_dist:
                    min_dist = d
                    min_s = Cj

            map.append(min_s)

        return map