from alg import Instance
from util import minkowski_distance

import numpy as np
import matplotlib.pyplot as plt

class ToyDatasets():
    def __init__(self, csv_output_file, img_folder):
        self.data_points = []
        self.data_labels = []
        self.instances = []
        
        self.csv_output_file = csv_output_file
        self.img_folder = img_folder

    def add_dataset(self, points, labels, k, p=2, img_name=''):
        'Adds a sklearn toy dataset and adds a corresponding instance'

        dist = self.make_dist_matrix(points, p)
        #Make labels
        S = list(range(len(points)))
        inst =  Instance(S, k, dist)

        self.data_points.append(points)
        self.data_labels.append(labels)
        self.instances.append(inst)

        if img_name:
            self.plot_dataset_img(len(self.instances) - 1, img_name)


    def make_dist_matrix(self, points, p):
        'Returns a distance matrix based on dataset points and p value for minkowski distance'

        #Can sacrifice one liner for performance assuming points are geometric, only calculate half the points
        return np.array([np.array([minkowski_distance(p1,p2,p) for p2 in points]) for p1 in points])

    def plot_dataset_img(self, idx, img_name):
        'Plots the dataset in index idx as a .pdf file with path img_folder/img_name'

        labels = set(self.data_labels[idx])
        num_labels = len(labels)
        label_to_color = {label: i for i, label in enumerate(labels)}
    
        # Map the labels to color indices
        color_indices = [label_to_color[label] for label in self.data_labels[idx]]

        X = [x[0] for x in self.data_points[idx]]
        Y = [x[1] for x in self.data_points[idx]]

        cmap = plt.get_cmap('tab20', num_labels)

        plt.scatter(X,Y, cmap=cmap, c=color_indices)
        
        plt.savefig(f"{self.img_folder}/{img_name}", format='pdf')

