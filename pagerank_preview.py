from src.utils_for_personalization import create_sites_matrix_for_personalization
from src.elasticsearch_interaction import create_sites_matrix, get_all_pages
from src.pagerank import sparse_matrix, pagerank_power
from elasticsearch import Elasticsearch

import numpy as np
import json
import os


PREVIEW_MODE = True
PERSONALIZATION_MODE = False
ENCODING = "utf-8"
SAMPLE_DATA = os.path.join("files", "sample_data.json")
RESULTS_PATH = os.path.join("files", "result_domain_ranks.json")
PERSONALIZATION_RESULTS_PATH = os.path.join("files", "personalized_result_domain_ranks.json")
LINKS_PATH = os.path.join("files", "all_links.json")
PERSONALIZATION = ["facebook", "mit", "slader", "youtube", "geeksforgeeks", "python", "linkedin", "developer"]


def read_json(json_path):
    with open(json_path, "r", encoding=ENCODING) as json_file:
        json_dict = json.load(json_file)
    return json_dict


def get_pers_vector(personalization):
    links_dict = read_json(LINKS_PATH)

    pers_indicies = []
    for n_domain, domain in enumerate(links_dict):
        domain = domain[8:]

        # check if this domain is personalizable
        for pers_subdomain in personalization:
            if pers_subdomain in domain:
                pers_indicies.append(n_domain)
                break

    # create personalized vector
    pers_vector = [0.0 for _ in range(len(links_dict))]
    transition_prob = 1 / len(pers_indicies)

    for idx in pers_indicies:
        pers_vector[idx] = transition_prob
    return np.array(pers_vector)


def assign_pagerank(result_ranks):
    links_dict = read_json(LINKS_PATH)

    result_dict = dict()
    for n_domain, domain in enumerate(links_dict):
        result_dict[domain] = result_ranks[n_domain]

    result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    if PERSONALIZATION_MODE:
        save_path = PERSONALIZATION_RESULTS_PATH

    else:
        save_path = RESULTS_PATH

    with open(save_path, "w", encoding=ENCODING) as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # prepare data
    if not PREVIEW_MODE:
        es_ = Elasticsearch([os.environ['ELASTICSEARCH_URL']],
                            http_auth=(os.environ['USERNAME'], os.environ['PASSWORD']))
        all_pages_ = get_all_pages(os.environ["INDEX_ELASTIC_COLLECTED_DATA"], es_)

        print("\n\n all_pages created !!!")

        # create matrix of edges among domain-nodes
        edges_, dim = create_sites_matrix(all_pages_['hits']['hits'])
    else:
        all_pages_ = read_json(SAMPLE_DATA)
        edges_, dim = create_sites_matrix_for_personalization(all_pages_['hits']['hits'])

    # create CSR matrix
    len_edges = len(edges_)
    weights_ = [1 / len_edges for _ in range(len_edges)]
    sls_ = sparse_matrix(weights_, edges_, dim)

    if PERSONALIZATION_MODE:
        # apply personalization
        pers_vector_ = get_pers_vector(PERSONALIZATION)

        # calculate and asign ranks
        result_ranks_ = pagerank_power(sls_, personalize=pers_vector_)

    else:
        result_ranks_ = pagerank_power(sls_)

    assign_pagerank(result_ranks_)
