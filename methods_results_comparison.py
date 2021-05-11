import json

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


def get_random_graph(
        min_size=20,
        max_size=2000,
        min_density=0.1,
        max_density=0.5):
    """ Creates a random graph and a teleport vector and output them
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

    """
    # G_size = random.randint(min_size, max_size)
    G_size = 500
    # G_size = 1533
    # p = random.uniform(min_density, max_density)
    p = 0.3

    A = sp.sparse.random(G_size, G_size, p, format='csr')
    nxG = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data

    personalize_vector = np.random.random(G_size)
    personalize_dict = dict(enumerate(personalize_vector.reshape(-1)))
    return A, nxG, iG, personalize_vector, personalize_dict


if __name__ == '__main__':
    print("started work")

    damping_factor = 0.85
    tol = 1e-3

    A, nxG, iG, personalize_vector, personalize_dict = get_random_graph()
    sys.stdout.flush()

    # TODO
    result_pagerank_numpy = nx.pagerank_numpy(nxG, alpha=damping_factor)
    result_pagerank_scipy = nx.pagerank_scipy(nxG, alpha=damping_factor, tol=tol)
    result_personalized_pagerank = iG.personalized_pagerank(
        directed=True,
        damping=damping_factor,
        weights=iG.es['weight'],
        implementation="prpack"
    )

    files_names = ["pagerank_numpy_domain_ranks", "pagerank_scipy_domain_ranks",
                   "personalized_pagerank_domain_ranks"]
    # for j, result_ranks in enumerate([result_pagerank_numpy, result_pagerank_scipy,
    #                                          result_personalized_pagerank]):
    #     # result_dict = dict()
    #     # result_dict[files_names[j]] =
    #     if j == 2:
    #         result_dict = dict()
    #         len_result_ranks = len(result_ranks)
    #         for k in range(len_result_ranks):
    #             result_dict[k] = result_ranks[k]
    #
    #     else:
    #         result_dict = result_ranks
    #
    #     # result_dict = {k: v for k, v in
    #     #                sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}
    #
    #     with open("result_" + files_names[j] + ".json", "w", encoding="utf-8") as f:
    #         json.dump(result_dict, f, indent=4, ensure_ascii=False)

    # create json from array
    new_result_personalized_pagerank = dict()
    len_result_ranks = len(result_personalized_pagerank)
    for k in range(len_result_ranks):
        new_result_personalized_pagerank[k] = result_personalized_pagerank[k]

    result_fast_pagerank = pagerank_power(A)
    result_fast_pagerank_dict = dict()
    for n_domain, domain in enumerate(result_fast_pagerank):
        # print(result_ranks[n_domain], "  --  ", domain)
        result_fast_pagerank_dict[n_domain] = result_fast_pagerank[n_domain]

    # with open("result_fast_pagerank.json", "w", encoding="utf-8") as f:
    #     json.dump(result_fast_pagerank_dict, f, indent=4, ensure_ascii=False)



    result_ranks = [result_fast_pagerank_dict, result_pagerank_numpy,
                    result_pagerank_scipy, new_result_personalized_pagerank]

    # len_result_ranks = len(result_ranks)
    # for k in range(len_result_ranks):
    #     result_ranks[k] = {k: v for k, v in sorted(result_ranks[k].items(),
    #                                            key=lambda item: item[1], reverse=True)}

    # with open("result_fast_pagerank_domain_ranks.json", "w", encoding="utf-8") as f:
    #     json.dump(result_dict, f, indent=4, ensure_ascii=False)

    columns_names = ['Domain ID', 'Our Fast Pagerank Value', 'Numpy Pagerank Value',
                     'Scipy Pagerank Value']
    df = pd.DataFrame(columns=columns_names)
    n_df_rows = len(result_ranks[0])
    len_columns_names = len(columns_names)

    for k in range(n_df_rows):
        new_row = [k]

        for m in range(len_columns_names - 1):
            if result_ranks[m].get(k, -1) != -1:
                new_row.append(result_ranks[m][k])
            # else:
            #     print("result_ranks[m] -- ", result_ranks[m])

        if len(new_row) == len_columns_names:
            df.loc[k] = new_row

    df.to_csv(os.path.join("files", "comparison_domains_ranks.csv"), index=False, header=True)
