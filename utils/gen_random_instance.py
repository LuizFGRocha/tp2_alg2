import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from scipy.stats import multivariate_normal

def parse_tuple(s):
    try:
        items = s.strip('()').split(',')
        return float(items[0]), float(items[1])
    except:
        raise argparse.ArgumentTypeError("Tuple must be in the form (float,float)")

def parse_float_list(s):
    try:
        return [float(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List must be a comma-separated list of floats")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('k', type=int)
    parser.add_argument('points_per_center', type=int)
    parser.add_argument('cov_list', type=parse_float_list, help="List of floats containing ")
    parser.add_argument('centers', nargs='+', type=parse_tuple, help="List of tuple containing centers coordinates")
    parser.add_argument('-s', '--show', action='store_true')

    args = parser.parse_args()

    if len(args.centers) != args.k:
        sys.exit("Number of centers must be the same as k")
    if len(args.cov_list) != 4:
        sys.exit("cov list must be of size 4")

    all_points = []

    with open(args.filename, 'w') as f:
        f.write(f"CENTERS:{args.k}\n")
        f.write(f"COV:{args.cov_list}\n")
        f.write(f"POINTS_PER_CENTER:{args.points_per_center}\n")
        f.write(f"CENTERS_COORDS:{args.centers}\n")
        f.write(f"POINTS/LABELS: \n")

        for idx, center in enumerate(args.centers):
            dist = multivariate_normal(center, np.array(args.cov_list).reshape((2,2)), allow_singular=True)

            points = dist.rvs(args.points_per_center)
            all_points += [(p, idx) for p in points]
            for p in points:
                f.write(f"{p[0]};{p[1]};{idx}\n")
    
    if args.show:
        fig, ax = plt.subplots()

        X = [p[0][0] for p in all_points]
        Y = [p[0][1] for p in all_points]
        label = [p[1] for p in all_points]

        cmap = plt.get_cmap('tab20', args.k)

        ax.scatter(X,Y, cmap=cmap, c=label)

        plt.show()

    

if __name__ == "__main__":
    main()