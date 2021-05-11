'''
Calcualate PageRank on several random graphs.
'''
import scipy as sp
import pandas as pd
import timeit
import os
import sys
import random
import igraph
import numpy as np
import networkx as nx

sys.path.insert(0, '..')
# from fast_pagerank.pagerank import pagerank
from pagerank import pagerank_power, sparse_matrix


# def print_and_flush(args):

#     sys.stdout.flush()
def get_random_graph(
        min_size=20,
        max_size=2000,
        min_density=0.1,
        max_density=0.5):
    ''' Creates a random graph and a teleport vector and output them
        in different formats for different algorithms

    Inputs
    ------

    min_size and max_size: The size of the graph will be a random number
        in the range of (min_size, max_size)
    min_sparsity and max_sparsity: The sparcity of the graph
        will be a random number in the range of (min_sparsity, max_sparsity)

    Returns
    -------

    nxG: A random Graph for NetworkX
    A: The equivallent csr Adjacency matrix, for our PageRank
    iG: The equivallent iGraph
    personalize_vector: Personalization probabily vector
    personalize_dict: Personalization probabily vector,
                    in the form of a dictionary for NetworkX

    '''
    G_size = random.randint(min_size, max_size)
    p = random.uniform(min_density, max_density)

    A = sp.sparse.random(G_size, G_size, p, format='csr')
    nxG = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data

    personalize_vector = sp.random.random(G_size)
    personalize_dict = dict(enumerate(personalize_vector.reshape(-1)))
    return A, nxG, iG, personalize_vector, personalize_dict


def get_real_data_graph(
        min_size=20,
        max_size=2000,
        min_density=0.1,
        max_density=0.5):
    G_size = random.randint(min_size, max_size)
    p = random.uniform(min_density, max_density)

    #     A = sp.sparse.random(G_size, G_size, p, format='csr')
    edges_ = np.load('edges.npy')
    len_edges = len(edges_)
    weights_ = [1 / len_edges for _ in range(len_edges)]
    matrix_dim = 43466
    A = sparse_matrix(weights_, edges_, matrix_dim)
    nxG = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data

    personalize_vector = sp.random.random(G_size)
    personalize_dict = dict(enumerate(personalize_vector.reshape(-1)))
    return A, nxG, iG, personalize_vector, personalize_dict


if __name__ == '__main__':
    print("started work")
    n = 5
    number_of_graphs = 1

    node_size_vector = sp.zeros(number_of_graphs)
    edge_size_vector = sp.zeros(number_of_graphs)
    # netx_pagerank_times = sp.zeros(number_of_graphs)
    netx_pagerank_times_numpy = sp.zeros(number_of_graphs)
    netx_pagerank_times_scipy = sp.zeros(number_of_graphs)
    ig_pagerank_times = sp.zeros(number_of_graphs)
    pagerank_times = sp.zeros(number_of_graphs)
    pagerank_times_power = sp.zeros(number_of_graphs)

    damping_factor = 0.85
    tol = 1e-3

    for i in range(number_of_graphs):
        #     A, nxG, iG, personalize_vector, personalize_dict = get_random_graph()
        A, nxG, iG, personalize_vector, personalize_dict = get_real_data_graph()
        node_size_vector[i] = A.shape[0]
        edge_size_vector[i] = A.count_nonzero()
        print("Graph %d: Nodes: %d, Edges: %d ..." % (i, node_size_vector[i], edge_size_vector[i]))
        sys.stdout.flush()

        #     networkx.pagerank commented out, because it is too slow

        #     netx_pagerank_times[i] = timeit.timeit(
        #         lambda: nx.pagerank(nxG, alpha=damping_factor, tol=tol),
        #         number=n) / n

        netx_pagerank_times_numpy[i] = timeit.timeit(
            lambda: nx.pagerank_numpy(nxG, alpha=damping_factor),
            number=n) / n

        netx_pagerank_times_scipy[i] = timeit.timeit(
            lambda: nx.pagerank_scipy(nxG, alpha=damping_factor, tol=tol),
            number=n) / n

        # iGraph, only "prpack", which is their latest version.
        ig_pagerank_times[i] = timeit.timeit(
            lambda: iG.personalized_pagerank(directed=True,
                                             damping=damping_factor,
                                             weights=iG.es['weight'],
                                             implementation="prpack"),
            number=n) / n

        #     My implementations

        # pagerank_times[i] = timeit.timeit(
        #     lambda: pagerank(A, p=damping_factor),
        #     number=n) / n
        pagerank_times_power[i] = timeit.timeit(
            # lambda: pagerank_power(A, p=damping_factor, tol=tol),
            # number=n) / n
            lambda: pagerank_power(A),
            number=n) / n

    argsort = edge_size_vector.argsort()

    edge_size_vector_sorted = edge_size_vector[argsort]
    node_size_vector_sorted = node_size_vector[argsort]

    # netx_pagerank_times_sorted = netx_pagerank_times[argsort]
    netx_pagerank_times_numpy_sorted = netx_pagerank_times_numpy[argsort]
    netx_pagerank_times_scipy_sorted = netx_pagerank_times_scipy[argsort]

    ig_pagerank_times_sorted = ig_pagerank_times[argsort]

    pagerank_times_sorted = pagerank_times[argsort]
    pagerank_times_power_sorted = pagerank_times_power[argsort]

    comparison_table = pd.DataFrame(list(zip(node_size_vector_sorted,
                                             edge_size_vector_sorted,
                                             #                                          netx_pagerank_times_sorted,
                                             netx_pagerank_times_numpy_sorted,
                                             netx_pagerank_times_scipy_sorted,
                                             ig_pagerank_times_sorted,
                                             pagerank_times_sorted,
                                             pagerank_times_power_sorted)),
                                    columns=['Nodes', 'Edges',
                                             #                                          'NetX',
                                             'NetX (numpy)',
                                             'NetX (scipy)',
                                             'iGraph',
                                             '(fast) pagerank',
                                             '(fast) pagerank_power']). \
        astype({'Nodes': 'int32', 'Edges': 'int32'})
    comparison_table.to_csv('pagerank_methods_comparison.csv')
    print("Done")
