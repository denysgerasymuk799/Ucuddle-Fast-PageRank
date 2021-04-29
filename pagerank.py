import json

import numpy as np
import scipy.sparse as sparse

from elasticsearch_interaction import create_sites_matrix

ROWS = 1
ROW_FORM = -1

STANDARD_RW_PROB = 0.85
MAX_ITER_NUM = 500
DELIMITER = 1e-06


def sparse_matrix(weights, edges, node_num):
    """
    The CSR Matrix is basically the 3 arrays:
    1. Value Array - is an array of all non-zero values of a standard matrix.
    2. Column Array - is an array that represents a column of a certain value.
    3. Row Array - is an array that represents a row of a certain value.
    * (0, 1) 0.4923 * represents a values 0.4923 in 0 row and 1 column.
    """
    return sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(node_num, node_num))


def diagonal_outdegree_matrix(a, dim):
    """
    Calculate diagonal out-degree matrix D.

    :param a: (scipy.sparse.csr.csr_matrix) a csr graph (sparse matrix)
    :param dim: (int) - dimension of a matrix
    :return: (scipy.sparse.csr.csr_matrix) - D,
             (numpy.ndarray) - the vector of row-wise sum of the elements of A
    """
    # sum all rows of a sparse matrix
    row_sums = np.asarray(a.sum(axis=ROWS)).reshape(ROW_FORM)

    # get the array of indices of elements that are non-zero
    k = row_sums.nonzero()[0]

    # create the diagonal matrix of the outdegree of each node in A
    d = sparse.csr_matrix((1 / row_sums[k], (k, k)), shape=(dim, dim))
    return d, row_sums


def pagerank_power(a, prob=STANDARD_RW_PROB, max_iter=MAX_ITER_NUM, tol=DELIMITER, personalize=None):
    """
    Calculate PageRank given a CSR Graph.

    :param a: (scipy.sparse.csr.csr_matrix) a csr graph (sparse matrix)
    :param prob: (float) - probability that random walker follows the link
    :param max_iter: (int) - max number of iterations
    :param tol: (float) - delimiter that stops the pagerank procedure
    :param personalize: (numpy.ndarray) - personalized probability distributions
    :return: (numpy.ndarray) - page ranks
    """
    dim = a.shape[0]    # dimension of a matrix
    d, row_wise_sum = diagonal_outdegree_matrix(a, dim)

    if not personalize:
        # generate a probability distribution of 1 (default value)
        personalize = np.ones(dim)
    personalize = personalize.reshape(dim, 1)

    # add an edge from every dangling node to every other node j with a weight of s_j
    s = (personalize / personalize.sum()) * dim

    # [np.newaxis, :] - make z_T be a row vector (a.k.a. T)
    z_t = (((1 - prob) * (row_wise_sum != 0) + (row_wise_sum == 0)) / dim)[np.newaxis, :]

    w = prob * a.T @ d

    x = s  # initially x = s because all we have for now is the personalized preferences
    old_x = np.zeros((dim, 1))

    iters = 0
    while np.linalg.norm(x - old_x) > tol and iters < max_iter:
        old_x = x
        x = w @ x + s @ (z_t @ x)
        iters += 1
    x = x / sum(x)

    return x.reshape(ROW_FORM)


if __name__ == '__main__':
    edges_, matrix_dim = create_sites_matrix()
    print("edges created")

    len_edges = len(edges_)
    weights_ = [1 / len_edges for _ in range(len_edges)]

    sls = sparse_matrix(weights_, edges_, matrix_dim)

    result_ranks = pagerank_power(sls)
    print("Result ranks of sites -- ", result_ranks)

    with open("all_links.json", "r", encoding="utf-8") as f:
        links_dict = json.load(f)

    result_dict = dict()
    for n_domain, domain in enumerate(links_dict):
        # print(result_ranks[n_domain], "  --  ", domain)
        result_dict[domain] = result_ranks[n_domain]

    result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    with open("result_domain_ranks.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
