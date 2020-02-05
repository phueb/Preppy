from typing import List, Set

import numpy as np


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for w in tokens:
        res[-1].append(w)
        if w in punctuation:
            res.append([])
    return res


def make_windows_mat(
        part: List[int],
        num_windows: int,
        num_tokens_in_window: int,
        ) -> np.ndarray:
    """
    return a matrix, where rows are windows.
    each window is an ordered array of word IDs.
    windows are created by sliding a moving window across tokens, moving one token at a time.
    """
    result = np.zeros((num_windows, num_tokens_in_window), dtype=np.int)
    for window_id in range(num_windows):
        window = part[window_id:window_id + num_tokens_in_window]
        result[window_id, :] = window
    return result