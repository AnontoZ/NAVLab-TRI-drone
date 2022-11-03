import matplotlib.pyplot as plt
import numpy as np
def plot_relative_distances(rel_dists, filepath):

    fig = plt.figure()
    plt.plot(rel_dists[:, 0], rel_dists[:, 1])
    plt.savefig(filepath, dpi=fig.dpi)
    plt.show()

def plot_locations(pos1, pos2):
    plt.plot(pos1[:, 0], pos1[:, 1])
    plt.plot(pos2[:, 0], pos2[:, 1])
    plt.show()



def plot_relative_distances_and_gt(rel_dists, ground_truth, ratios_1, ratios_2, loc1, loc2, filepath):

    fig = plt.figure()
    plt.plot(rel_dists[:, 0], rel_dists[:, 1], label="estimate")
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], label="gt")
    plt.legend()
    plt.savefig(filepath, dpi=fig.dpi)
    plt.figure()
    norm_errors = np.linalg.norm(rel_dists-ground_truth, axis=1)
    plt.plot(norm_errors)
    plt.title("Error")
    plt.figure()
    plt.plot(ratios_1, label="Car 1")
    plt.plot(ratios_2, label="Car 2")
    plt.legend()
    plt.figure()


    car_dists = np.linalg.norm(ground_truth, axis=1)
    plt.plot(car_dists, label="distances")
    plt.title("Distance between cars")

    plt.figure()
    indices = np.argsort(car_dists)
    plt.plot(car_dists[indices], norm_errors[indices])
    plt.title("Car distance vs. error")


    loc_dists = np.linalg.norm(loc1, axis=1)+np.linalg.norm(loc2, axis=1)
    plt.figure()
    indices = np.argsort(loc_dists)
    plt.plot(loc_dists[indices].flatten(), norm_errors[indices].flatten())
    plt.title("Distance from image center vs. error")
    plt.show()