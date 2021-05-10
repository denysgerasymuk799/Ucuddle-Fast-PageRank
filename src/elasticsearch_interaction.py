from src.utils import reduce_to_domain

import numpy as np
import json


MIN_LINK_LEN = 11


def get_all_pages(index_name, es):
    # TODO: Note the size param, which increases the hits displayed from the default (10) to 1000 per shard
    res = es.search(index=index_name, body={
        "query": {
            "match_all": {}
        },
        "size": 1000
    })

    print("Got %d Hits" % len(res['hits']['hits']))
    return res['hits']['hits']


def create_links_dict(all_pages):
    links_dict = dict()

    n_link = 0
    for n_site, site in enumerate(all_pages):
        link = site["_source"]["link"]
        link = reduce_to_domain(link)

        if len(link) >= MIN_LINK_LEN and links_dict.get(link, -1) == -1:
            links_dict[link] = n_link
            n_link += 1

        if site["_source"]["hyperlinks"] is None:
            continue

        for child_link in site["_source"]["hyperlinks"]:
            child_link = reduce_to_domain(child_link)

            if len(child_link) >= MIN_LINK_LEN and links_dict.get(child_link, -1) == -1:
                links_dict[child_link] = n_link
                n_link += 1

    with open("files/all_links.json", "w", encoding="utf-8") as f:
        json.dump(links_dict, f, indent=4, ensure_ascii=False)


def create_sites_matrix(all_pages):
    """
    Create numpy matrix of edges among domain-nodes
    """
    create_links_dict(all_pages)

    with open("files/all_links.json", "r", encoding="utf-8") as f:
        links_dict = json.load(f)

    pages_matrix = np.array([[]])
    for i, page in enumerate(all_pages):
        link = reduce_to_domain(page["_source"]["link"])
        if len(link) < MIN_LINK_LEN:
            continue

        n_page = links_dict[link]

        for j, child_link in enumerate(page["_source"]["hyperlinks"]):
            child_link = reduce_to_domain(child_link)
            if len(child_link) < MIN_LINK_LEN:
                continue

            if i == 0 and j == 0:
                pages_matrix = np.array([[n_page, links_dict[child_link]]])
            else:
                if n_page != links_dict[child_link]:
                    pages_matrix = np.append(pages_matrix, [[n_page, links_dict[child_link]]], axis=0)

    return pages_matrix, len(links_dict)
