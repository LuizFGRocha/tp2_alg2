import numpy as np

def minkowski_distance(x,y,p=2):
    return np.sum(np.abs(x - y) ** p) ** (1/p)

def read_custom_synthetic(filename):
    points = []
    labels = []
    centers = -1

    in_points = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if in_points:
                #Parse point
                x,y,lab = line.split(";")
                points.append(np.array([float(x), float(y)]))
                labels.append(int(lab))
                continue
            
            label, val = line.split(":")
            if label == "CENTERS":
                centers = int(val)
            elif label == "POINTS/LABELS":
                in_points = True

    return np.array(points), np.array(labels), centers

def read_real(filename):
    points = []
    labels = []
    centers = -1

    in_points = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if in_points:
                #Parse point
                values = line.split(";")
                labels.append(int(values.pop()))
                point = [float(x) for x in values]
                points.append(np.array(point))
                continue
            
            label, val = line.split(":")
            if label == "CENTERS":
                centers = int(val)
            elif label == "POINTS/LABELS":
                in_points = True

    return np.array(points), np.array(labels), centers