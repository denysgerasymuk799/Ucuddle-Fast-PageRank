def find_nth_character(find_char, string, num_occurrences):
    len_string = len(string)
    for i in range(len_string):
        if string[i] == find_char:
            num_occurrences -= 1

            if num_occurrences == 0:
                return i

    return -1


def reduce_to_domain(link):
    end_substring_pos = find_nth_character("/", link, 3)
    if end_substring_pos != -1:
        link = link[:end_substring_pos]

    return link.lower()
