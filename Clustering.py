import random

from sklearn import datasets, manifold
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets, manifold
from sklearn.decomposition import PCA

from sklearn.metrics import pairwise


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return circles
    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """

    # euclid_matrix = []
    # for first in range(len(X)):
    #     first_distance_vac = []
    #     for sec in range(len(Y)):
    #         dist = np.linalg.norm(X[first] - Y[sec])
    #         first_distance_vac.append(dist)
    #     euclid_matrix.append(first_distance_vac)
    # return euclid_matrix

    euclid_matrix = pairwise.euclidean_distances(X, Y)
    return euclid_matrix



def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    # C = sum(X) / len(X)
    # return C
    return np.average(X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    rand_idx = np.random.choice(len(X), 1)[0]
    centroids = [X[rand_idx]]
    indexes_chosen = rand_idx
    relevent_data = X
    while k > len(centroids):
        relevent_data = np.delete(relevent_data, indexes_chosen, axis=0)
        metric_dist = np.min(metric(relevent_data, centroids), axis=1)

        prob = np.square(metric_dist) / np.square(metric_dist).sum()
        cumprobs = prob.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        centroids.append(X[ind])
        indexes_chosen = ind
    return np.reshape(centroids, (k, centroids[0].shape[0]))


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    centroids = init(X, k, metric)
    for _ in range(iterations):
        dist = metric(centroids, X)
        min_x = np.argmin(dist, axis=0)
        for i in range(k):
            i_cent_lst = [X[m] for m in range(X.shape[0]) if min_x[m] == i]
            centroids[i] = center(i_cent_lst)
    return min_x, centroids



def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(- np.power(X, 2) / (np.power(sigma, 2) * 2))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    sort_matrix = np.matrix.argsort(X)[:, 1:m+1]
    similarity_matrix = np.zeros(X.shape)
    for i in range(len(X)):
        similarity_matrix[i][sort_matrix[i]] = 1
    return similarity_matrix


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    n = X.shape[0]
    S = euclid(X, X)
    W = similarity(S, similarity_param)
    D = np.diag(np.sum(W, axis=1))
    # D = np.zeros(S.shape)
    # for i in range(n):
    #     D[i][i] = np.sum(W[i])
    D_sqrt_inv = np.linalg.pinv(np.sqrt(D))
    L = np.identity(n) - D_sqrt_inv.dot(W.dot(D_sqrt_inv))
    eigen_values, eigen_vectors = np.linalg.eigh(L)
    sort_eigen_values = np.argsort(abs(eigen_values))[:k]
    corresponding_eigen_vectors = eigen_vectors[:, sort_eigen_values]
    kmeans_run = kmeans(corresponding_eigen_vectors, k)
    return kmeans_run


def elbow(X, max_k=10, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The elbow method. all parameters but max_k are the same as in the kmeans function.
    plots the cost function as a function of k.
    :param max_k: maximum k we will check.
    """
    sum_of_squared_distance = []
    for k in range(1, max_k + 1):
        clusters_list, centroids = kmeans(X, k, iterations, metric, center, init)
        sum = np.sum(np.min(metric(X, centroids), axis=1)) / X.shape[0]
        sum_of_squared_distance.append(sum)
    plt.plot([i for i in range(1, max_k + 1)], sum_of_squared_distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum of squared distance')
    plt.title('Elbow Method')
    plt.show()


def biological_clustering(kmns, clust):
    """
    plots the biological data.
    """
    data_path = 'microarray_data.pickle'
    with open(data_path, 'rb') as f:
        biological_clustering = pickle.load(f)
    num_of_clust = clust

    data = biological_clustering
    if kmns:
        clusters, centroids = kmeans(data, num_of_clust)
    else:
        clusters, centroids = spectral(data, num_of_clust, 3)
    for i in range(num_of_clust):
        cur_centroid_data = data[np.where(clusters == i)]
        plt.figure()
        plt.imshow(cur_centroid_data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
        plt.colorbar()
        plt.title('number of samples in cluster= ' + str(np.shape(cur_centroid_data)[0]))
        plt.show()



def plot_func(kmns, clust, data, sigma=3, s=gaussian_kernel):
    """
    plots the kmeans and the spectral.
    """
    num_of_clust = clust
    if kmns:
        clusters, centroids = kmeans(data, num_of_clust)
    else:
        clusters, centroids = spectral(data, num_of_clust, sigma, s)
    centroid_data = []
    colors = ["blue", "red", "green", "yellow", "grey", "purple", "orange", "pink", "black", "salmon"]
    plt.figure()
    for i in range(num_of_clust):
        cur_centroid_data = data[np.where(clusters == i)]
        centroid_data.append(cur_centroid_data)
        plt.plot(centroid_data[i][:, 0], centroid_data[i][:, 1], '.', color=colors[i])
    if kmns:
        plt.title("K-MEANS")
    else:
        plt.title("Spectral for sigma = " + str(sigma))
    plt.show()

def gaussians_check():
    """
    returns four gaussians.
    """
    gauss_a = np.random.normal(loc=[5, 5], size=(120, 2))
    gauss_b = np.random.normal(loc=[5, -5], size=(120, 2))
    gauss_c = np.random.normal(loc=[-5, 5], size=(120, 2))
    gauss_d = np.random.normal(loc=[-5, -5], size=(120, 2))

    gaussians = np.concatenate((gauss_a, gauss_b, gauss_c, gauss_d))
    return np.random.permutation(gaussians)


def tsne_vs_pca():
    """
    plot the t-SNE and PCA, and find out which one is better.
    """
    digits = datasets.load_digits(n_class=9)
    data = digits.data
    tSNE = manifold.TSNE(n_components=2)
    data_tsne = tSNE.fit_transform(data)
    min, max = np.min(data_tsne, 0), np.max(data_tsne, 0)
    d = (data_tsne - min) / (max - min)

    colors = ["grey", "Purple", "Orange", "Pink", "black", "blue", "red", "green", "yellow"]
    plt.figure()
    for i in range(len(d)):
        label = digits.target[i]
        plt.text(d[i, 0], d[i, 1], str(label), color=colors[label])

    plt.title("t - SNE algorithm")
    plt.yticks([])
    plt.xticks([])
    plt.show()


    pca_algo = PCA(n_components=2)
    data_pca = pca_algo.fit_transform(data)
    min, max = np.min(data_pca, 0), np.max(data_pca, 0)
    d = (data_pca - min) / (max - min)
    plt.figure()
    for i in range(d.shape[0]):
        label = digits.target[i]
        plt.text(d[i, 0], d[i, 1], str(label), color=colors[label])

    plt.title("PCA algorithm")
    plt.yticks([])
    plt.xticks([])
    plt.show()
    plt.figure()


def plot_similarities(data_set, clust, sigma):
    """
    plots the weights matrix.
    """
    gaussian_ker = gaussian_kernel(euclid(data_set, data_set), sigma)
    plt.imshow(gaussian_ker)
    plt.title("pre cluster")
    plt.show()

    spec_clust = spectral(data_set, clust, sigma, gaussian_kernel)
    spec_clust = np.argsort(spec_clust[0])
    gaussian_ker = gaussian_kernel(euclid(data_set[spec_clust], data_set[spec_clust]), sigma)
    plt.imshow(gaussian_ker)
    plt.title("post cluster")
    plt.show()


if __name__ == '__main__':
    data_path = 'microarray_data.pickle'
    with open(data_path, 'rb') as f:
        micro = pickle.load(f)

    circles = np.asarray(circles_example()).T

    path = 'APML_pic.pickle'
    with open(path, 'rb') as f:
        apml = pickle.load(f)

