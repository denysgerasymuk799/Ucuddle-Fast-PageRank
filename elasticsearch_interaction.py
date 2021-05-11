import math
import os
import json
import time

import numpy as np

from elasticsearch import Elasticsearch
from multiprocessing import Pool, Lock
from pprint import pprint

from elasticsearch.exceptions import TransportError

from utils import reduce_to_domain

# es = Elasticsearch([os.environ['ELASTICSEARCH_URL']],
#                    http_auth=(os.environ['Username'], os.environ['Password']))

es = Elasticsearch(["http://77.120.203.142:9200"],
                   http_auth=("ucuddle", "ucuddleCloud123)"))

MIN_LINK_LEN = 11
# N_PROCESSES = int(os.environ["N_PROCESSES"])
N_PROCESSES = 12
# mutex = Lock()


def parallel_get_pages(args):
    n_requests, from_id, step, index_name = args
    all_sites_arr = []
    for _ in range(n_requests):
        # TODO: Note the size param, which increases the hits displayed from the default (10) to 1000 per shard
        waiting_response_time = 0
        for i in range(5):
            time.sleep(waiting_response_time)

            try:
                res = es.search(
                    index=index_name,
                    body={
                        "from": from_id,
                        "query": {
                            "match_all": {}
                        },
                        "size": step,
                        "sort": {
                            "site_id": "asc"
                        }
                    },
                    request_timeout=1000
                )
                print("Got %d Hits" % len(res['hits']['hits']))
                # all_sites_arr = np.append(all_sites_arr, res['hits']['hits'])

                for site in res['hits']['hits']:
                    all_sites_arr.append({
                        "link": site["_source"]["link"],
                        "hyperlinks": site["_source"]["hyperlinks"]
                    })

                break
            except TransportError as exc:
                print('index setup error', exc)

            waiting_response_time = math.exp(i + 1)

        from_id += step
        time.sleep(10)

        # with mutex:
        #     all_sites_arr = np.append(all_sites_arr, res['hits']['hits'], axis=0)

    return all_sites_arr


def get_all_pages(index_name):
    step = int(os.environ["N_TAKEN_SITES_PER_REQUEST"])
    n_requests = int(os.environ["N_REQUESTS"])

    n_processes = N_PROCESSES
    div_requests = n_requests // n_processes
    mod_requests = n_requests % n_processes
    requests_per_process = [div_requests] * n_processes
    for i in range(mod_requests):
        requests_per_process[i] += 1

    print("requests_per_process -- ", requests_per_process)

    from_id = 0
    processes_data = []
    for i in range(n_processes):
        processes_data.append([requests_per_process[i], from_id, step, index_name])
        from_id += requests_per_process[i] * step

    print("processes_data -- ", processes_data)
    with Pool(n_processes) as pool:
        all_sites_arr = pool.map(parallel_get_pages, processes_data)

    return all_sites_arr


def create_links_dict(all_pages):
    links_dict = dict()

    n_link = 0
    for j in range(N_PROCESSES):
        for n_site, site in enumerate(all_pages[j]):
            # link = site["_source"]["link"]
            link = site["link"]
            link = reduce_to_domain(link)

            if len(link) >= MIN_LINK_LEN and links_dict.get(link, -1) == -1:
                links_dict[link] = n_link
                n_link += 1

            if site["hyperlinks"] is None:
                continue

            for child_link in site["hyperlinks"]:
                child_link = reduce_to_domain(child_link)

                if len(child_link) >= MIN_LINK_LEN and links_dict.get(child_link, -1) == -1:
                    links_dict[child_link] = n_link
                    n_link += 1

    with open(os.path.join("files", "all_links.json"), "w", encoding="utf-8") as f:
        json.dump(links_dict, f, indent=4, ensure_ascii=False)


def create_sites_matrix():
    """

    Create numpy matrix of edges among domain-nodes
    """
    all_pages = get_all_pages(os.environ["INDEX_ELASTIC_COLLECTED_DATA"])
    print("\n\n !!! all_pages created")

    # print("all_pages -- ", all_pages)
    # for i, site in enumerate(all_pages):
    #     print(site["_source"]["link"])
    #     print(site["_source"]["hyperlinks"])
    #     print("\n\n\n")

    # answer = input("Do you want to overwrite 'all_links.json' ? -- ")
    # if answer == "yes":
    create_links_dict(all_pages)

    with open(os.path.join("files", "all_links.json"), "r", encoding="utf-8") as f:
        links_dict = json.load(f)

    pages_matrix = np.array([[]])
    for j in range(N_PROCESSES):
        for i, page in enumerate(all_pages[j]):
            link = reduce_to_domain(page["link"])
            if len(link) < MIN_LINK_LEN:
                continue

            n_page = links_dict[link]

            for j, child_link in enumerate(page["hyperlinks"]):
                child_link = reduce_to_domain(child_link)
                if len(child_link) < MIN_LINK_LEN:
                    continue

                if pages_matrix.size == 0:
                    pages_matrix = np.array([[n_page, links_dict[child_link]]])
                else:
                    if n_page != links_dict[child_link]:
                        # print("links_dict[child_link] -- ", links_dict[child_link])
                        # print("pages_matrix -- ", pages_matrix)
                        pages_matrix = np.append(pages_matrix, [[n_page, links_dict[child_link]]], axis=0)

    # for i in range(len(pages_matrix)):
    #     print(pages_matrix[i])
    return pages_matrix, len(links_dict)


if __name__ == '__main__':
    # all_pages = get_all_pages(os.environ["INDEX_ELASTIC_COLLECTED_DATA"])
    # create_links_dict(all_pages)
    # create_sites_matrix()

    result = get_all_pages("production_data1")
    print(result)
    print(len(result))

    create_links_dict(result)
