from pagerank import pagerank_power

import scipy as sp
import pandas as pd
import os
import igraph
import networkx as nx


def get_random_graph():
    """
    Creates a random graph and a teleport vector and output them
        in different formats for different algorithms

    Inputs
    ------

    min_size and max_size: The size of the graph will be a random number
        in the range of (min_size, max_size)
    min_sparsity and max_sparsity: The sparcity of the graph
        will be a random number in the range of (min_sparsity, max_sparsity)

    Returns
    -------

    nx_graph: A random Graph for NetworkX
    A: The equivallent csr Adjacency matrix, for our PageRank
    iG: The equivallent iGraph
    personalize_vector: Personalization probabily vector
    personalize_dict: Personalization probabily vector,
                    in the form of a dictionary for NetworkX

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
    # sys.stdout.flush()

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
    # for j, result_ranks in enumerate([result_pagerank_numpy, result_pagerank_scipy,
    #                                          result_pagerank_igraph]):
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
    new_result_pagerank_igraph = dict()
    len_result_ranks = len(result_pagerank_igraph)
    for k in range(len_result_ranks):
        new_result_pagerank_igraph[k] = result_pagerank_igraph[k]

    result_fast_pagerank = pagerank_power(A)
    result_fast_pagerank_dict = dict()
    for n_domain, domain in enumerate(result_fast_pagerank):
        # print(result_ranks[n_domain], "  --  ", domain)
        result_fast_pagerank_dict[n_domain] = result_fast_pagerank[n_domain]

    # with open("result_fast_pagerank.json", "w", encoding="utf-8") as f:
    #     json.dump(result_fast_pagerank_dict, f, indent=4, ensure_ascii=False)

    result_ranks = [result_fast_pagerank_dict, result_pagerank_numpy,
                    result_pagerank_scipy, new_result_pagerank_igraph]

    # len_result_ranks = len(result_ranks)
    # for k in range(len_result_ranks):
    #     result_ranks[k] = {k: v for k, v in sorted(result_ranks[k].items(),
    #                                            key=lambda item: item[1], reverse=True)}

    # with open("result_fast_pagerank_domain_ranks.json", "w", encoding="utf-8") as f:
    #     json.dump(result_dict, f, indent=4, ensure_ascii=False)

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
