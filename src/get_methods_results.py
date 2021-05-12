from pagerank import pagerank_power

import scipy as sp
import pandas as pd
import os
import igraph
import networkx as nx


def get_random_graph():
    """
    Creates a random graph, which is similar to real website network of relations
    """
    graph_size = 500
    p = 0.3

    A = sp.sparse.random(graph_size, graph_size, p, format='csr')
    nx_graph = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nx_graph.edges()), directed=True)
    iG.es['weight'] = A.data

    return A, nx_graph, iG


if __name__ == '__main__':
    print("Work started")

    damping_factor = 0.85
    tol = 1e-3

    A, nx_graph, iG = get_random_graph()

    result_pagerank_numpy = nx.pagerank_numpy(nx_graph, alpha=damping_factor)
    result_pagerank_scipy = nx.pagerank_scipy(nx_graph, alpha=damping_factor, tol=tol)
    result_pagerank_igraph = iG.pagerank(
        directed=True,
        damping=damping_factor,
        weights=iG.es['weight'],
        implementation="prpack"
    )

    files_names = ["pagerank_numpy_domain_ranks", "pagerank_scipy_domain_ranks",
                   "pagerank_igraph_domain_ranks"]

    # create json from array
    new_result_pagerank_igraph = dict()
    len_result_ranks = len(result_pagerank_igraph)
    for k in range(len_result_ranks):
        new_result_pagerank_igraph[k] = result_pagerank_igraph[k]

    result_fast_pagerank = pagerank_power(A)
    result_fast_pagerank_dict = dict()
    for n_domain, domain in enumerate(result_fast_pagerank):
        result_fast_pagerank_dict[n_domain] = result_fast_pagerank[n_domain]

    result_ranks = [result_fast_pagerank_dict, result_pagerank_numpy,
                    result_pagerank_scipy, new_result_pagerank_igraph]

    columns_names = ['Domain ID', 'Our Fast PR Value', 'NetworkX Numpy PR Value',
                     'NetworkX Scipy PR Value', 'IGraph PR Value']
    df = pd.DataFrame(columns=columns_names)
    n_df_rows = len(result_ranks[0])
    len_columns_names = len(columns_names)

    for k in range(n_df_rows):
        new_row = [k]

        for m in range(len_columns_names - 1):
            if result_ranks[m].get(k, -1) != -1:
                new_row.append(result_ranks[m][k])

        if len(new_row) == len_columns_names:
            df.loc[k] = new_row

    df.to_csv(os.path.join("files", "comparison_domains_ranks.csv"), index=False, header=True)
