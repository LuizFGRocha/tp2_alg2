from alg import Instance
from util import minkowski_distance
from sklearn.metrics import silhouette_score, adjusted_rand_score
from time import time

import numpy as np
import matplotlib.pyplot as plt

class ToyDatasets():
    def __init__(self, csv_output_file, img_folder):
        self.data_points = []
        self.data_labels = []
        self.instances = []
        self.names = []
        
        self.csv_output_file = csv_output_file
        self.img_folder = img_folder

        #Write .csv header
        with open(csv_output_file, 'w') as f:
            f.write("instance,radius,silhouette,adj_rand_score,exec_time,method\n")

    def add_dataset(self, points, labels, k, name, p=2):
        'Adds a sklearn toy dataset and adds a corresponding instance'

        dist = self.make_dist_matrix(points, p)
        #Make labels
        S = list(range(len(points)))
        inst =  Instance(S, k, dist)

        self.data_points.append(points)
        self.data_labels.append(labels)
        self.instances.append(inst)
        self.names.append(name)

        self.plot_dataset_img(len(self.instances) - 1)


    def make_dist_matrix(self, points, p):
        'Returns a distance matrix based on dataset points and p value for minkowski distance'

        #Can sacrifice one liner for performance assuming points are geometric, only calculate half the points
        return np.array([np.array([minkowski_distance(p1,p2,p) for p2 in points]) for p1 in points])

    def plot_dataset_img(self, idx, labels=None, img_name='', circles=None, title=''):
        'Plots the dataset in index idx as a .pdf file with path img_folder/img_name'

        fig, ax = plt.subplots()

        if labels is None:
            labels = self.data_labels[idx]

        set_labels = set(labels)
        label_to_color = {label: i for i, label in enumerate(set_labels)}
    
        # Map the labels to color indices
        color_indices = [label_to_color[label] for label in labels]

        X = [x[0] for x in self.data_points[idx]]
        Y = [x[1] for x in self.data_points[idx]]

        cmap = plt.get_cmap('tab20', len(set_labels))

        ax.scatter(X,Y, cmap=cmap, c=color_indices)

        if circles is not None:
            #Also plot circles
            for origin, radius, lab in circles:
                circ = plt.Circle(origin, radius, color=cmap(label_to_color[lab]), fill=False)
                ax.scatter([origin[0]], [origin[1]], c='r')
                ax.add_patch(circ)

        ax.set_aspect('equal', 'box')
        fig.suptitle(title)
        plt.savefig(f"{self.img_folder}/{self.names[idx] if img_name == '' else img_name}.pdf", format='pdf')

    def test_datasets(self, itr=30):
        'Test all datasets for itr iterations'
        for idx, (inst, name) in enumerate(zip(self.instances, self.names)):
            for _ in range(itr):
                start = time()
                C = inst.k_clusters()
                exec_time = time() - start
                labels, radius = inst.cluster_map(C)
                
                #Calculate associated circles
                circles = list(zip([self.data_points[idx][p] for p in C], radius, C))

                #Plot result
                self.plot_dataset_img(idx, labels, img_name=self.names[idx] + 'result', 
                                      circles=circles, title=f"Max radius: {max(radius)}")

                #Write results
                self.write_results(idx, C, labels, radius, exec_time, "a")

    def write_results(self, idx, C, labels, radius, exec_time, method):
        'Writes relevant results for the csv'
        max_radius = max(radius)
        #TODO ver melhor como funciona essas duas metricas
        sil = silhouette_score(self.instances[idx].dist, labels)
        #adjrand = adjusted_rand_score()
        adjrand = None
        

        with open(self.csv_output_file, 'w') as f:
            #f.write("instance,radius,silhouette,adj_rand_score,exec_time,method\n")
            f.write(f"{self.names[idx]},{max_radius},{sil},{adjrand},{exec_time},{method}")