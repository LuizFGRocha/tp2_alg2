from alg import Instance
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from util import build_dist_matrix, time_execution
import multiprocessing as mp

#This must be out of the original class to call from pool
def single_test_and_write(inst, og_labels, method):
        (C,labels,radius), exec_time = time_execution(method)
        sil = silhouette_score(inst.dist, labels)
        adjrand = adjusted_rand_score(og_labels,labels)

        return (C,labels,radius), exec_time, (sil, adjrand)

class ToyDatasets():
    def __init__(self, csv_output_file, img_folder):
        self.curr_points = []
        self.curr_labels = []
        self.curr_instance = None
        self.curr_name = ''
        self.pool = mp.Pool()
        
        self.csv_output_file = csv_output_file
        self.img_folder = img_folder

        #Write .csv header
        with open(csv_output_file, 'w') as f:
            f.write("instance,radius,silhouette,adj_rand_score,exec_time,method,p\n")

    def test_dataset(self, points, labels, k, name, p=2, itr=30):
        'Creates an instance of a dataset and runs it for testing'

        dist = self.make_dist_matrix(points, p)
        #Make point names
        S = list(range(len(points)))
        inst =  Instance(S, k, points, p, dist)

        self.curr_points = points
        self.curr_labels = labels
        self.curr_instance = inst
        self.curr_name = name

        self.plot_dataset_img()

        self.test_curr_dataset(itr)

    def make_dist_matrix(self, points, p):
        'Returns a distance matrix based on dataset points and p value for minkowski distance'

        return build_dist_matrix(points, p)

    def plot_dataset_img(self, labels=None, img_name='', circles=None, title=''):
        'Plots the current dataset as a .pdf file with path img_folder/img_name'

        fig, ax = plt.subplots()

        if labels is None:
            labels = self.curr_labels

        set_labels = set(labels)
        label_to_color = {label: i for i, label in enumerate(set_labels)}
    
        # Map the labels to color indices
        color_indices = [label_to_color[label] for label in labels]

        X = [x[0] for x in self.curr_points]
        Y = [x[1] for x in self.curr_points]

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
        plt.savefig(f"{self.img_folder}/{self.curr_name if img_name == '' else img_name}.pdf", format='pdf')
        plt.close()

    def test_curr_dataset(self, itr=30):
        'Test current dataset for itr iterations'
        inst = self.curr_instance
        for method, method_name in zip([inst.scikit_k_clusters, inst.k_clusters, inst.refining_k_clusters], ["scikit", "greedy", "refining"]):
            best_C, best_labels, radiuses, best_r = None, None, None, np.inf
            
            results = self.pool.starmap(single_test_and_write, [(inst, self.curr_labels, method)] * itr)

            #Write results
            for (C,labels,radius), exec_time, (sil, adjrand) in results:
                self.write_results(self.curr_name, sil, adjrand, max(radius), exec_time, method_name, self.curr_instance.p)

            #Get best result
            best_r_ind = np.argmin([max(r) for (_,_,r), _, _ in results])

            (best_C, best_labels, radiuses), _, _ = results[best_r_ind]
            best_r = max(radiuses)
            
            #Plot best result
            #Calculate associated circles
            circles = list(zip(best_C, radiuses))
            self.plot_dataset_img(best_labels, img_name=self.curr_name + method_name, 
                                    circles=circles, title=f"Max radius: {best_r}")

    def write_results(self, name, sil, adjrand, radius, exec_time, method, p):
        'Writes relevant results for the csv'
        
        with open(self.csv_output_file, 'a') as f:
            f.write(f"{name},{float(radius)},{float(sil)},{float(adjrand)},{exec_time},{method},{p}\n")