from typing import List, Any
from itertools import combinations


def get_all_combinations(data_list: List[Any]) -> List[Any]:
    all_combinations = []
    for i in range(len(data_list) + 1, 1, -1):
        all_combinations += list(combinations(data_list, i))
    return all_combinations
