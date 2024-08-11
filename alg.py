import numpy as np
from sklearn.cluster import KMeans
from util import minkowski_distance

# S: conjunto de pontos
# C: resultado, pontos que representam centros
# dist[Si, Sj]: distância entre Si e Sj, armazenada em uma matriz de distancias
# Queremos minimizar o raio dos clusters: min max dist(Si, Cj)

class Instance:
    def __init__(self, S, k, X, p, dist_matrix):
        self.S = S
        self.k = k
        self.X = X
        self.p = p
        self.dist = dist_matrix

        self.KMeans = KMeans(n_clusters=k)

    def k_clusters_r(self, max_r):
        'Solves K-means using max radius 2-approximation algorithm'
        S = list(np.random.permutation(self.S)) # Permute S for different results, convert to list for pop
        if self.k >= len(S):
            return S

        C = []

        while len(S) > 0:
            s = S.pop()
            C.append(s)

            for Si in list(S):
                if self.dist[Si, s] < 2 * max_r:
                    S.remove(Si)

        if len(C) > self.k:
            return None
        
        labels, radius = self.cluster_map(C)
        
        #Update C with positions from actual
        C = [self.X[c] for c in C]

        return C, labels, radius

    def k_clusters(self):
        'Solves K-means through greedy 2-approximation algorithm'
        S = list(np.random.permutation(self.S)) #Convert to list for pop()
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
        
        labels, radius = self.cluster_map(C)

        #Update C with actual positions
        C = [self.X[c] for c in C]

        return C, labels, radius
    
    def refining_k_clusters(self):
        max_r = 0

        # Finds max radius
        for Si in self.S:
            for Sj in self.S:
                if self.dist[Si, Sj] > max_r:
                    max_r = self.dist[Si, Sj]

        return self.recursive_k_clusters(0, max_r)
    
    def recursive_k_clusters(self, lower_bound, upper_bound, epsilon=1e-6):
        if upper_bound - lower_bound < epsilon:
            return self.k_clusters_r(upper_bound)

        r = (upper_bound + lower_bound) / 2
        C = self.k_clusters_r(r)

        if C is None:
            return self.recursive_k_clusters(r, upper_bound)
        else:
            return self.recursive_k_clusters(lower_bound, r)
        
    def scikit_k_clusters(self):
        'Solves k-means using scikit implementation'
        
        kmeans = self.KMeans.fit(self.X)

        #Need to calculate radius for each cluster
        radius = [0] * len(kmeans.cluster_centers_)

        for x in self.X:
            min_dist = np.inf
            min_idx = None

            for idx, Cj in enumerate(kmeans.cluster_centers_):
                d = minkowski_distance(x, Cj, self.p)
                if d < min_dist:
                    min_dist = d
                    min_idx = idx
            
            if min_dist > radius[min_idx]:
                radius[min_idx] = min_dist
        
        return kmeans.cluster_centers_, kmeans.labels_, radius

    def cluster_map(self, C):
        'Maps each element from S to a cluster in C, according to its lowest distance.'
        map = []
        radius = [0] * len(C) #Start radius as 0

        for Si in self.S:
            min_dist = np.inf
            min_idx = None

            for idx, Cj in enumerate(C):
                d = self.dist[Si, Cj]
                if d < min_dist:
                    min_dist = d
                    min_idx = idx
        
            if min_dist > radius[min_idx]:
                radius[min_idx] = min_dist

            map.append(min_idx)

        return map, radius