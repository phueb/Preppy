"""
the legacy Prep classes are for compatibility with older projects (e.g. StartingSmall).
they contain logic for handling multiple iterations over ordered language data.
multiple iterations were used in 2019 master's thesis of PH.

"""

from typing import List, Optional, Generator, Tuple
from cached_property import cached_property
import numpy as np
import random
from functools import reduce
from operator import iconcat
from numpy.lib.stride_tricks import as_strided

from preppy.tokenstore import TokenStore
from preppy.util import split_into_sentences, make_windows_mat


class PartitionedPrep:
    """
    generates windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    text is split into  partitions.
    this means that input must already be ordered as desired in the text file.
    """
    def __init__(self,
                 docs: List[str],
                 reverse: bool,
                 num_types: Optional[int],  # can be None
                 num_parts: int,
                 num_iterations: Tuple[int, int],
                 batch_size: int,
                 context_size: int,
                 num_evaluations: int = 10,
                 shuffle_within_part: bool = False,
                 ):
        """
        A document has type string. It is not tokenized.
        Tokenization happens here - using string.split()

        only input docs from train split.
        use SlidingPrep for docs from test split
        """

        # make store
        tokenized_docs = [d.split() for d in docs]
        tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
        self.store = TokenStore(tokens, num_parts, batch_size, context_size, num_types)

        self.reverse = reverse
        self.num_types = num_types or len(self.store.types)
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
    def num_iterations_list(self) -> List[int]:
        return list(np.linspace(self.num_iterations[0], self.num_iterations[1],
                                num=self.num_parts, dtype=np.int))

    @property
    def mean_num_iterations(self) -> int:
        result = np.mean(self.num_iterations_list)
        assert float(result).is_integer()
        return int(result)

    @property
    def num_mbs_in_token_ids(self) -> int:
        return self.num_mbs_in_part * self.num_parts

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

    # /////////////////////////////////////////////////////////////////// mbs

    @property
    def num_mbs_in_part(self) -> int:
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @property
    def num_mbs_in_block(self) -> int:
        """a block is a partition that has been repeated num_iterations times"""
        return self.num_mbs_in_part * self.mean_num_iterations

    @property
    def num_mbs(self) -> int:
        """number of total mini-batches considering iterations"""
        return self.num_mbs_in_block * self.num_parts

    @cached_property
    def num_mbs_per_eval(self) -> int:
        res = self.num_mbs / self.num_evaluations
        assert float(res).is_integer()
        return int(res)

    @cached_property
    def eval_mbs(self) -> List[int]:
        """mini-batches at which evaluation should take place"""
        end = self.num_mbs_per_eval * self.num_evaluations + self.num_mbs_per_eval
        eval_mbs = list(range(0, end, self.num_mbs_per_eval))
        return eval_mbs

    @cached_property
    def stop_mb(self) -> int:
        """
        this is where training actually stops.
        it is not guaranteed to be the last mb, due to equally dividing mbs into eval_mbs
        """
        res = self.eval_mbs[-1]
        assert res == self.num_mbs, (res, self.num_mbs)
        return res

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

        # is very memory efficient as it operates on views of original data
        res = as_strided(np.array(self.store.token_ids, dtype=np.int64),  # do not change d-type without strides
                         shape=(self.num_parts, self.num_tokens_in_part),
                         strides=(8 * self.num_tokens_in_part, 8),
                         writeable=False)
        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res

    @cached_property
    def reordered_windows(self) -> np.ndarray:
        """
        not used during training, but is useful for offline analysis of data
        """
        num_possible_windows = len(self.store.token_ids) - self.num_tokens_in_window
        res = as_strided(np.array(self.store.token_ids, dtype=np.int64),
                         shape=(num_possible_windows, self.num_tokens_in_window),
                         strides=(8, 8), writeable=False)
        print(f'Matrix containing all windows has shape={res.shape}')

        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res
        
    def generate_batches(self):
        """alias of gen_windows()"""
        yield from self.gen_windows()

    def gen_windows(self,
                    iterate_once: Optional[bool] = False,
                    reordered_parts: Optional[List[List[int]]] = None
                    ) -> Generator[np.ndarray, None, None]:
        """yield from 3d array where each 2d slice is a batch of windows with shape (batch_size, context size + 1)"""

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


