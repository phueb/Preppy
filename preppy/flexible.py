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
from itertools import chain, repeat
from operator import iconcat
from numpy.lib.stride_tricks import as_strided

from preppy.tokenstore import TokenStore
from preppy.utils import split_into_sentences


class FlexiblePrep:
    """
    generate batches containing windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    chose to slide over tokens using sliding batch, or iterate over partitions of tokens.
    """
    def __init__(self,
                 docs: List[str],
                 reverse: bool,
                 sliding: bool,
                 num_types: Optional[int] = None,  # can be None
                 num_parts: int = 2,
                 num_iterations: Tuple[int, int] = (8, 32),
                 batch_size: int = 64,
                 context_size: int = 7,
                 num_evaluations: int = 8,
                 shuffle_within_part: bool = False,
                 ):

        assert num_evaluations % num_iterations[0] == 0  # not very important
        assert num_evaluations % num_iterations[1] == 0  # not very important

        # make store
        tokenized_docs = [d.split() for d in docs]
        tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
        self.store = TokenStore(tokens, num_parts, batch_size, context_size, num_types)

        self.reverse = reverse
        self.sliding = sliding
        self.num_types = num_types or len(self.store.types)
        self.num_parts = num_parts
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_evaluations = num_evaluations
        self.shuffle_within_part = shuffle_within_part

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
        return self.num_tokens_in_part - self.context_size

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
        num_possible_windows = len(self.store.token_ids) - self.context_size
        res = as_strided(np.array(self.store.token_ids, dtype=np.int64),
                         shape=(num_possible_windows, self.num_tokens_in_window),
                         strides=(8, 8), writeable=False)
        print(f'Matrix containing all windows has shape={res.shape}')

        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res

    def batches_from_strides(self,
                             part: List[int],
                             num_iterations: int,
                             ) -> np.ndarray:
        """return 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size)."""
        stride = 8  # 8 bytes in 64 bits

        if self.sliding:
            num_possible_mbs = self.num_windows_in_part - self.batch_size + 1
            num_requested_mbs = self.num_mbs_in_part * num_iterations

            # get all possible unique mbs in part
            slide_size = 1
            _batches = as_strided(np.array(part, dtype=np.int64),
                                  shape=(num_possible_mbs, self.batch_size, self.num_tokens_in_window),
                                  strides=(stride * slide_size, stride, stride),
                                  writeable=False)

            # get exactly the number of requested mbs form all possible mbs, keeping order
            assert num_requested_mbs <= num_possible_mbs
            num_remaining = num_possible_mbs - num_requested_mbs
            bool_ids = np.random.permutation([True] * num_requested_mbs + [False] * num_remaining)
            batches = _batches[bool_ids]
        else:
            num_requested_mbs = self.num_mbs_in_part
            slide_size = self.batch_size
            batches = as_strided(np.array(part, dtype=np.int64),
                                 shape=(num_requested_mbs, self.batch_size, self.num_tokens_in_window),
                                 strides=(stride * slide_size, stride, stride),
                                 writeable=False)

        if self.reverse:
            return np.flip(batches, axis=0)
        else:
            return batches

    def generate_batches(self,
                         verbose: bool = False,
                         iterate_once: bool = False,
                         reordered_parts: Optional[List[List[int]]] = None
                         ) -> Generator[np.ndarray, None, None]:
        """
        yield from 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size).
        a window is an array of token IDs for context words, a target word and the next-word.
        """

        if iterate_once:  # useful for computing perplexity on train split
            num_iterations_list = [1] * self.num_parts
        else:
            num_iterations_list = self.num_iterations_list

        if reordered_parts is None:
            reordered_parts = self.reordered_parts

        for part_id, part in enumerate(reordered_parts):
            num_iterations = num_iterations_list[part_id]

            if verbose:
                print(f'part_id={part_id}')
                print(part)

            # get batches by sliding incrementally across tokens
            if self.sliding:
                batches = self.batches_from_strides(part, num_iterations)
                yield from batches

            # get batches by iterating over ordered partitions of tokens
            else:
                batches = self.batches_from_strides(part, num_iterations)
                yield from chain.from_iterable(repeat(tuple(batches), num_iterations))


