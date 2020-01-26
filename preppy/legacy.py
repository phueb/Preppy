"""
the legacy Prep classes are for compatibility with older projects (e.g. StartingSmall).
they contain logic for handling multiple iterations over ordered language data.
multiple iterations were used in 2019 master's thesis of PH.

"""

from typing import List, Optional, Generator, Union, Set
from cached_property import cached_property
import numpy as np
import random
from functools import reduce
from operator import iconcat
from numpy.lib.stride_tricks import as_strided

from preppy.tokenstore import TokenStore


class TrainPrep:
    """
    generates windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    text is split into 2 partitions - more then 2 partitions is not supported.
    this means that input must already be ordered as desired in the text file.
    """
    def __init__(self,
                 docs: List[str],
                 reverse: bool,
                 num_types: Union[int, None],
                 num_parts: int,
                 num_iterations: List[int],
                 batch_size: int,
                 context_size: int,
                 num_evaluations: int,
                 shuffle_within_part: Optional[bool] = False,
                 ):
        """
        A document has type string. It is not tokenized.
        Tokenization happens here - using string.split()

        only input docs from train split.
        use TestPrep for docs from test split
        """

        # make store
        tokenized_docs = [d.split() for d in docs]
        tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
        self.store = TokenStore(tokens, num_parts, batch_size, context_size, num_types)

        self.reverse = reverse
        self.num_types = num_types
        self.num_parts = num_parts
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_evaluations = num_evaluations
        self.shuffle_within_part = shuffle_within_part

        self.token_ids_array = np.array(self.store.token_ids, dtype=np.int64)
        if self.token_ids_array.dtype == np.int64:
            self.stride = 8  # bytes because 64 bits = 2 bytes ; changing this may cause CUDA error
        else:
            raise ValueError('Stride must be changed if data-type is changed')

    @property
    def num_mbs_in_part(self) -> int:
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @property
    def num_iterations_list(self) -> List[int]:
        return list(np.linspace(self.num_iterations[0], self.num_iterations[1],
                                num=self.num_parts, dtype=np.int))

    @property
    def mean_num_iterations(self) -> int:
        result = np.mean(self.num_iterations_list)
        assert float(result).is_integer()
        return int(result)

    @property
    def num_mbs_in_block(self) -> int:
        """
        a block is a partition that has been repeated num_iterations times
        """
        return self.num_mbs_in_part * self.mean_num_iterations

    @property
    def num_tokens_in_window(self) -> int:
        return self.context_size + 1

    @property
    def num_windows_in_part(self) -> int:
        return self.num_tokens_in_part - self.num_tokens_in_window

    @property
    def num_tokens_in_part(self) -> int:
        result = self.store.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    @property
    def num_mbs(self) -> int:
        return self.num_mbs_in_part * self.num_parts

    # /////////////////////////////////////////////////////////////////// mbs

    @cached_property
    def num_mbs_per_eval(self):
        return self.num_mbs // self.num_evaluations

    @cached_property
    def eval_mbs(self) -> List[int]:
        """
        mini-batches at which evaluation should take place
        """
        end = self.num_mbs_per_eval * self.num_evaluations + self.num_mbs_per_eval
        eval_mbs = list(range(0, end, self.num_mbs_per_eval))
        return eval_mbs

    @cached_property
    def stop_mb(self):
        """
        this is where training actually stops.
        it is not guaranteed to be the last mb, due to equally dividing mbs into eval_mbs
        """
        return self.eval_mbs[-1]

    # /////////////////////////////////////////////////////////////////// parts & windows

    @cached_property
    def midpoint(self) -> int:
        res = self.store.num_tokens // 2
        assert res * 2 == self.store.num_tokens
        return res

    @cached_property
    def reordered_parts(self) -> np.ndarray:

        # shuffle sentences within each half.
        # this eliminates any effect of cycling (iterating multiple times on data in the same order)
        if self.shuffle_within_part:
            print('Shuffling within part')
            half1 = self.store.tokens[:self.midpoint]
            half2 = self.store.tokens[self.midpoint:]
            # shuffle sentences (not tokens)
            sentences1 = split_into_sentences(half1, punctuation={'.', '!', '?'})
            sentences2 = split_into_sentences(half2, punctuation={'.', '!', '?'})
            random.shuffle(sentences1)
            random.shuffle(sentences2)
            # flatten sentences and overwrite existing tokens
            tokens = reduce(iconcat, sentences1, []) + reduce(iconcat, sentences2, [])
            self.store.set_tokens(tokens)

        # strided is very memory efficient as it operates on views of original data
        strided = as_strided(np.array(self.store.token_ids, dtype=np.int16),  # do not change d-type without strides
                             shape=(self.num_parts, self.num_tokens_in_part),
                             strides=(2 * self.num_tokens_in_part, 2),
                             writeable=False)
        if self.reverse:
            return strided[::-1]
        else:
            return strided

    @cached_property
    def reordered_windows(self) -> np.ndarray:
        """
        not used during training, but is useful for offline analysis of data
        """
        num_possible_windows = len(self.token_ids_array) - self.num_tokens_in_window
        shape = (num_possible_windows, self.num_tokens_in_window)
        res = as_strided(self.token_ids_array, shape, strides=(self.stride, self.stride), writeable=False)
        print(f'Matrix containing all windows has shape={res.shape}')

        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res

    def gen_windows(self,
                    iterate_once: Optional[bool] = False,
                    reordered_parts: Optional[List[List[int]]] = None
                    ) -> Generator[np.ndarray, None, None]:
        """
        generates windows with structure [context, target] where contex is a list of word IDs and target is a word ID.
        """

        if iterate_once:  # useful for computing perplexity on train split
            num_iterations_list = [1] * self.num_parts
        else:
            num_iterations_list = self.num_iterations_list

        if reordered_parts is None:
            reordered_parts = self.reordered_parts

        # generate
        for part_id, part in enumerate(reordered_parts):
            windows_mat = make_windows_mat(part, self.num_windows_in_part, self.num_tokens_in_window)
            num_iterations = num_iterations_list[part_id]
            if not iterate_once:
                print('Iterating {} times over part {}'.format(num_iterations, part_id))

            for _ in range(num_iterations):
                for windows in np.vsplit(windows_mat, self.num_mbs_in_part):
                    yield windows


class TestPrep:
    """
    generates windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    text is not split into any partitions - it is treated like a single partition.
    """
    def __init__(self,
                 docs: List[str],
                 batch_size: int,
                 context_size: int,
                 vocab: List[str],
                 ):
        """
        only input docs from train split.
        use TestPrep for docs from test split
        """

        self.num_parts = 1

        # make store
        tokenized_docs = [d.split() for d in docs]
        tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
        num_types = len(vocab)
        self.store = TokenStore(tokens, self.num_parts, batch_size, context_size, num_types, vocab)

        self.num_types = num_types
        self.batch_size = batch_size
        self.context_size = context_size

    @cached_property
    def num_mbs_in_part(self) -> int:
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_token_ids(self) -> int:
        return self.num_mbs_in_part * self.num_parts

    @cached_property
    def num_tokens_in_window(self) -> int:
        return self.context_size + 1

    @cached_property
    def num_windows_in_part(self) -> int:
        return self.num_tokens_in_part - self.num_tokens_in_window

    @cached_property
    def num_tokens_in_part(self) -> int:
        result = self.store.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    # /////////////////////////////////////////////////////////////////// windows

    def gen_windows(self, iterate_once=True) -> Generator[np.ndarray, None, None]:
        """
        this was previously called "gen_ids", and it generated x, y rather than windows
        """

        del iterate_once  # it is a function argument to keep API consistent with TrainPrep

        part = self.store.token_ids  # no need to partition the test split

        # generate
        windows_mat = make_windows_mat(part, self.num_windows_in_part, self.num_tokens_in_window)
        for windows in np.vsplit(windows_mat, self.num_mbs_in_part):
            yield windows


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