from alg import Instance
from util import minkowski_distance

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
            f.write("instance,radius,silhouette,adj_rand_score,exec_time,method")

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

    def plot_dataset_img(self, idx):
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
        
        plt.savefig(f"{self.img_folder}/{self.names[idx]}.pdf", format='pdf')

    def test_datasets(self, itr=30):
        'Test all datasets for itr iterations'
        for it in range(itr):
            #TODO Run algo, record to csv
            pass