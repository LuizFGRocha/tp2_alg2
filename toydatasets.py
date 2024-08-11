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
        inst =  Instance(S, k, points, p, dist)

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
            for idx, (origin, radius) in enumerate(circles):
                circ = plt.Circle(origin, radius, color=cmap(label_to_color[idx]), fill=False)
                ax.scatter([origin[0]], [origin[1]], c='r')
                ax.add_patch(circ)

        ax.set_aspect('equal', 'box')
        fig.suptitle(title)
        plt.savefig(f"{self.img_folder}/{self.names[idx] if img_name == '' else img_name}.pdf", format='pdf')

    def time_execution(self, fn, *args):
        'Runs fn function with args, also returning execution time'
        start = time()
        res = fn(*args)
        exec_time = time() - start
                
        return res, exec_time

    def test_datasets(self, itr=30):
        'Test all datasets for itr iterations'
        for idx, (inst, name) in enumerate(zip(self.instances, self.names)):
            for method, method_name in zip([inst.scikit_k_clusters, inst.k_clusters, inst.refining_k_clusters], ["scikit", "greedy", "refining"]):
                best_C, best_labels, radiuses, best_r = None, None, None, np.inf
                for _ in range(itr):
                    (C, labels, radius), exec_time = self.time_execution(method)

                    max_r = max(radius)
                    if max_r < best_r:
                        best_C, best_labels, radiuses, best_r = C, labels, radius, max_r

                    #Write results
                    self.write_results(idx, C, labels, max_r, exec_time, method_name)
                
                #Plot best result
                #Calculate associated circles
                circles = list(zip(best_C, radiuses))
                self.plot_dataset_img(idx, best_labels, img_name=name + method_name, 
                                        circles=circles, title=f"Max radius: {best_r}")

    def write_results(self, idx, C, labels, radius, exec_time, method):
        'Writes relevant results for the csv'

        sil = silhouette_score(self.instances[idx].dist, labels)
        adjrand = adjusted_rand_score(self.data_labels[idx],labels)
        
        with open(self.csv_output_file, 'a') as f:
            f.write(f"{self.names[idx]},{float(radius)},{float(sil)},{float(adjrand)},{exec_time},{method}\n")