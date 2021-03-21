from typing import List
import numpy as np
import random
from functools import reduce
from operator import iconcat


def shuffle_at_sentence_level(tokens: List[str],
                              shuffle_seed: int = 20,
                              ) -> List[str]:
    """
    shuffle at sentence-level (as opposed to document-level)
    this remove clustering of same-age utterances within documents
    """

    # TODO sentences are detected with punctuation, but periods can occur in numbers, not just at boundaries
    # TODO: use a more sophisticated sentence boundary detector

    random.seed(shuffle_seed)
    print('WARNING: Shuffling sentences')
    sentences: List[List[str]] = split_into_sentences(tokens)
    random.shuffle(sentences)
    res = reduce(iconcat, sentences, [])  # flatten list of lists

    return res


def split_into_sentences(tokens: List[str],
                         ) -> List[List[str]]:

    res = [[]]
    for n, w in enumerate(tokens):
        res[-1].append(w)
        if w.endswith('.') or w.endswith('?') or w.endswith('!') and n < len(tokens) - 1:  # prevent  empty list at end
            res.append([])
    return res


def chunk_sentences(sentences: List[List[str]],
                    split_size: int,
                    ):
    for i in range(0, len(sentences), split_size):
        yield sentences[i:i + split_size]


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
