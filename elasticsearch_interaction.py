import os
import json
import numpy as np

from elasticsearch import Elasticsearch


es = Elasticsearch([os.environ['ELASTICSEARCH_URL']],
                    http_auth=(os.environ['USERNAME'], os.environ['PASSWORD']))


def get_all_pages(index_name):
    res = es.search(index=index_name, body={
        "query": {
            "match_all": {}
        },
        "size": 5
    })
    print("Got %d Hits" % res['hits']['total']['value'])
    return res['hits']['hits']


def create_links_dict(all_pages):
    links_dict = dict()

    n_link = 0
    for n_site, site in enumerate(all_pages):
        link = site["_source"]["link"]
        print(site["_source"]["hyperlinks"])
        if links_dict.get(link, -1) == -1:
            links_dict[link] = n_link
            n_link += 1

        for child_link in site["_source"]["hyperlinks"]:
            if links_dict.get(child_link, -1) == -1:
                links_dict[child_link] = n_link
                n_link += 1

    with open("all_links.json", "w", encoding="utf-8") as f:
        json.dump(links_dict, f, indent=4, ensure_ascii=False)


def create_sites_matrix():
    all_pages = get_all_pages("t_english_sites-a17")

    # create_links_dict(all_pages)

    with open("all_links.json", "r", encoding="utf-8") as f:
        links_dict = json.load(f)

    pages_matrix = np.array([[]])
    for i, page in enumerate(all_pages):
        n_page = links_dict[page["_source"]["link"]]
        # print(page["_source"]["hyperlinks"])

        for j, child_link in enumerate(page["_source"]["hyperlinks"]):
            if i == 0 and j == 0:
                pages_matrix = np.array([[n_page, links_dict[child_link]]])
            else:
                pages_matrix = np.append(pages_matrix, [[n_page, links_dict[child_link]]], axis=0)

    # for i in range(len(pages_matrix)):
    #     print(pages_matrix[i])


if __name__ == '__main__':
    create_sites_matrix()
