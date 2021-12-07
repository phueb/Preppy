"""
This Prep is an improved version of PartitionedPrep
because:
- it can generate batches of windows that slide incrementally over data
- handles both train and test tokens in one object
- prunes tokens more conservatively

"""

from typing import List, Optional, Generator, Tuple, Dict
import numpy as np
from itertools import chain, repeat
import itertools
from numpy.lib.stride_tricks import as_strided
import string
from sortedcontainers import SortedSet
from cached_property import cached_property


class Prep:
    """
    generate batches containing windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    chose to slide over tokens using sliding batch, or iterate over partitions of tokens.
    """

    def __init__(self,
                 tokens: List[str],
                 reverse: bool,
                 sliding: bool,
                 num_parts: int = 2,
                 num_iterations: Tuple[int, int] = (8, 32),
                 batch_size: int = 64,
                 context_size: int = 7,
                 shuffle_within_part: bool = False,
                 min_num_test_tokens: int = 0,
                 disallow_non_ascii: bool = False,
                 token2id: Optional[Dict[str, int]] = None,
                 ):

        if not isinstance(tokens, list):
            raise TypeError('Input to Prep must be a list of tokens')

        # check for non-ascii characters and new-lines
        for t in tokens:
            if '\n' == t:
                raise ValueError('Remove all newline characters before passing text to Prep')

            if disallow_non_ascii:
                for char in set(t):
                    if char != ' ' and char not in set(string.ascii_lowercase + string.punctuation + string.digits):
                        raise ValueError(f'Character "{char}" not allowed in Prep')

        self.reverse = reverse
        self.sliding = sliding
        self.num_parts = num_parts
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.context_size = context_size
        self.shuffle_within_part = shuffle_within_part
        self.min_num_test_tokens = min_num_test_tokens
        self._token2id = token2id  # can be overwritten by user

        # find num_tokens_train so that batching works
        # implementation note: find largest number of batches,
        # that can 1) fit into a part, and 2) leave enough tokens for testing.
        num_tokens_train = self.calc_num_tokens(tokens, self.num_parts, min_num_test_tokens)

        # get train tokens from start and end of corpus, so test tokens are from remaining middle tokens
        m = num_tokens_train // 2
        if num_tokens_train % 2 != 0:  # when num_tokens_train is not even, we must adjust by ading + 1
            self.tokens_train = tokens[:m + 1] + tokens[-m:]
            tokens_pruned = tokens[m+1:-m].copy()
        else:
            self.tokens_train = tokens[:m] + tokens[-m:]
            tokens_pruned = tokens[m:-m].copy()

        # checks
        if num_tokens_train < min_num_test_tokens:
            raise RuntimeError(f'Number of train tokens ({num_tokens_train:,}) is less '
                               f'than min_num_test_tokens ({min_num_test_tokens:,})')
        if len(self.tokens_train) != num_tokens_train:
            raise RuntimeError(f'Expected {num_tokens_train:,} tokens in training data, '
                               f'but there are {len(self.tokens_train):,}')
        if len(tokens_pruned) != len(tokens) - num_tokens_train:
            raise RuntimeError(f'Expected {len(tokens_pruned):,} pruned tokens, '
                               f'but there are {len(tokens) - num_tokens_train:,}')

        # note:
        # tokens_pruned is tokens available for test split
        # num_tokens_test_ is the smallest possible number of tokens that are compatible with batching + context_size,
        # given the number of tokens in tokens_pruned

        num_tokens_test_ = self.calc_num_tokens(tokens_pruned,
                                                num_parts=1,
                                                min_num_remaining_tokens=0)
        self.tokens_test_ = tokens_pruned[:num_tokens_test_]
        if len(tokens_pruned) < num_tokens_test_ < min_num_test_tokens:
            raise RuntimeError(f'Number of tokens needed to make batching + context_size work ({num_tokens_test_:,}) '
                               f'is larger than tokens available for test split ({len(tokens_pruned):,}). '
                               f'Consider increasing min_num_test_tokens to make more tokens available.')

        print(f'Num tokens in train={len(self.tokens_train):>9,}')
        print(f'Num tokens in test_={len(self.tokens_test_):>9,}')

    def calc_num_tokens(self,
                        _tokens,
                        num_parts: int,  # depends on train vs. test
                        min_num_remaining_tokens: int,
                        ):
        _num_tokens = 0
        tmp = []
        for num_batches in itertools.count():
            num_windows_in_part = self.batch_size * num_batches
            num_tokens_in_part = num_windows_in_part + self.context_size
            _num_tokens = num_tokens_in_part * num_parts  # how many tokens fit in part with given num batches

            if _num_tokens + min_num_remaining_tokens > len(_tokens):
                if tmp:
                    return tmp[-1]
                else:
                    return _num_tokens
            else:
                tmp.append(_num_tokens)

    # /////////////////////////////////////////////////////////////////// properties related to train tokens

    @cached_property
    def tokens(self):
        return self.tokens_train

    @cached_property
    def types(self) -> List[str]:
        return [t for t in SortedSet(self.tokens_train + self.tokens_test_)]

    @cached_property
    def num_types(self):
        return len(self.types)

    @cached_property
    def num_tokens(self):
        return len(self.tokens)

    @cached_property
    def token2id(self) -> Dict[str, int]:
        if self._token2id is not None:
            return self._token2id
        return {t: n for n, t in enumerate(self.types)}

    # /////////////////////////////////////////////////////////////////// basic properties for training (not testing)

    @property
    def num_iterations_list(self) -> List[int]:
        return list(np.linspace(self.num_iterations[0], self.num_iterations[1],
                                num=self.num_parts, dtype=np.int))

    @cached_property
    def mean_num_iterations(self) -> int:
        result = np.mean(self.num_iterations_list)
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_token_ids(self) -> int:
        return self.num_mbs_in_part * self.num_parts

    @cached_property
    def num_tokens_in_window(self) -> int:
        return self.context_size + 1

    @cached_property
    def num_windows_in_part(self) -> int:
        return self.num_tokens_in_part - self.context_size

    @cached_property
    def num_tokens_in_part(self) -> int:  # this is forced to be the same for train and test split
        result = self.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def midpoint(self) -> int:
        res = self.num_tokens // 2
        assert res * 2 == self.num_tokens
        return res

    # /////////////////////////////////////////////////////////////////// train mbs

    @cached_property
    def num_mbs_in_part(self) -> int:
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_block(self) -> int:
        """a block is a partition that has been repeated num_iterations times"""
        return self.num_mbs_in_part * self.mean_num_iterations

    @cached_property
    def num_mbs(self) -> int:
        """number of total mini-batches considering iterations"""
        return self.num_mbs_in_block * self.num_parts

    # /////////////////////////////////////////////////////////////////// parts & windows

    def get_reordered_parts(self,
                            tokens: List[str],
                            num_parts: int,
                            num_tokens_in_part: int,
                            ) -> np.ndarray:

        # shuffle sentences in each partition
        # this eliminates any effect of cycling (iterating multiple times on data in the same order)
        if self.shuffle_within_part:
            print('Shuffling within part')
            raise NotImplementedError  # TODO how to do this? use original sentences in self.sentences?

        else:
            tokens = tokens

        token_ids = [self.token2id[t] for t in tokens]

        # is very memory efficient as it operates on views of original data
        res = as_strided(np.array(token_ids, dtype=np.int64),  # do not change d-type without strides
                         shape=(num_parts, num_tokens_in_part),
                         strides=(8 * num_tokens_in_part, 8),
                         writeable=False)
        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res

    def batches_from_strides(self,
                             part: List[int],
                             num_iterations: int,
                             num_mbs_in_part: int,
                             ) -> np.ndarray:
        """return 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size)."""
        stride = 8  # 8 bytes in 64 bits

        if self.sliding:
            num_possible_mbs = self.num_windows_in_part - self.batch_size + 1
            num_requested_mbs = num_mbs_in_part * num_iterations

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
            num_requested_mbs = num_mbs_in_part
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
                         is_test: bool = False,
                         iterate_once: bool = False,
                         shuffle_at_start: bool = False,
                         ) -> Generator[np.ndarray, None, None]:
        """
        yield from 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size).
        a window is an array of token IDs for context words, a target word and the next-word.
        """

        if is_test:
            if self.min_num_test_tokens == 0:
                raise ValueError('Cannot generate batches when is_test=True and min_num_test_tokens=0')

        # we have to calculate sizes for test data because class properties are for train data only
        if is_test:
            print(f'Generating batches over {len(self.tokens_test_):,} test tokens', flush=True)
            tokens = self.tokens_test_
            num_parts = 1
            num_tokens_in_part = len(self.tokens_test_)
            num_mbs_in_part = int((num_tokens_in_part - self.context_size) / self.batch_size)
        else:
            tokens = self.tokens  # equivalent to tokens_train
            num_parts = self.num_parts
            num_tokens_in_part = self.num_tokens_in_part
            num_mbs_in_part = self.num_mbs_in_part

        if iterate_once or is_test:  # useful for computing perplexity on train split
            num_iterations_list = [1] * num_parts
        else:
            num_iterations_list = self.num_iterations_list

        reordered_parts = self.get_reordered_parts(tokens, num_parts, num_tokens_in_part)

        for part_id, part in enumerate(reordered_parts):
            num_iterations = num_iterations_list[part_id]

            batches = self.batches_from_strides(part, num_iterations, num_mbs_in_part)

            # check that there are no bad integers (from other memory locations) in the batch
            assert np.max(batches) <= self.num_types

            # shuffle batches WITHIN a partition
            if shuffle_at_start:
                batches = np.random.permutation(batches)

            # get batches by sliding incrementally across tokens
            if self.sliding:
                yield from batches
            # get batches by iterating over ordered partitions of tokens
            else:
                repeated = repeat(tuple(batches), num_iterations)
                yield from chain.from_iterable(repeated)

    # /////////////////////////////////////////////////////////////////// for analysis, not batching

    @cached_property
    def reordered_windows(self) -> np.ndarray:
        """
        not used during training, but is useful for offline analysis of data
        """
        token_ids = [self.token2id[t] for t in self.tokens]

        num_possible_windows = len(token_ids) - self.context_size
        res = as_strided(np.array(token_ids, dtype=np.int64),
                         shape=(num_possible_windows, self.num_tokens_in_window),
                         strides=(8, 8), writeable=False)
        print(f'Matrix containing all windows has shape={res.shape}')

        if self.reverse:
            return np.flip(res, axis=0)
        else:
            return res

    @cached_property
    def reordered_parts(self):
        return self.get_reordered_parts(self.tokens, self.num_parts, self.num_tokens_in_part)
