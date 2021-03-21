"""
PartitionedPrep is compatible with older projects (e.g. StartingSmall).
differs from FlexiblePrep:
- contains only logic for handling multiple iterations over ordered partitions, not sliding
- does not use huggingface tokenizers for tokenization and vectorisation

This class was used in 2019 master's thesis of Philip Huebner.

"""
from collections import OrderedDict, Counter
from itertools import islice
from typing import List, Optional, Generator, Tuple, Dict
from cached_property import cached_property
import numpy as np
import random
from functools import reduce
from operator import iconcat
from numpy.lib.stride_tricks import as_strided
from sortedcontainers import SortedSet

from preppy import configs
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
            sentences1 = split_into_sentences(half1)
            sentences2 = split_into_sentences(half2)
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


class TokenStore(object):
    """
    Prunes number of tokens to acceptable length for batching and partitioning.
    Removes out-of-vocabulary types
    """

    def __init__(self,
                 tokens: List[str],
                 num_parts: int,
                 batch_size: int,
                 context_size: int,
                 num_types: Optional[None],
                 _types: Optional[list] = None,  # pass a vocabulary when tokens originate in test split
                 oov: str = configs.Symbols.unk,
                 ):

        self.num_parts = num_parts
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_types = num_types
        self.oov = oov

        self._types = _types

        self.tokens_no_oov = self.prune(tokens)  # now tokens does not have to be stored in memory
        del tokens

    def make_pruning_length(self, num_raw) -> int:
        """
        Find length by which to prune tokens such that result is divisible by num_parts and
        such that the result of this division is divisible by batch_size
        after first subtracting num_words_in_window.
        One cannot simply add num_words_in_window*num_docs because this might give result that is
        larger than number of available tokens.
        One can't use num_words_in_window to calculate the factor, because num_words_in_window
        should only be added once only to each document
        """
        # factor
        num_parts = self.num_parts if self._types is None else 1
        factor = self.batch_size * (num_parts if self._types is None else 1) + self.context_size
        # make divisible
        num_factors = num_raw // factor
        adj = (num_factors - num_parts) * self.context_size
        result = num_factors * factor - adj
        return result

    def prune(self, raw: List[str]) -> List[str]:
        num_raw = len(raw)
        pruning_length = self.make_pruning_length(num_raw)
        pruned = raw[:pruning_length]
        print(f'Pruned {num_raw:,} total words to {pruning_length:,}')
        return pruned

    # /////////////////////////////////////////////////// properties

    @cached_property
    def w2f_no_oov(self) -> OrderedDict:
        c = Counter(self.tokens_no_oov)
        result = OrderedDict(
            sorted(c.items(), key=lambda item: (item[1], item[0]), reverse=True))  # order matters
        return result

    @cached_property
    def types(self) -> List[str]:
        if self._types is None:
            most_freq_words = list(islice(self.w2f_no_oov.keys(), 0, self.num_types))
            if self.num_types is not None:
                sorted_words = sorted(most_freq_words[:-1] + [self.oov])
            else:
                sorted_words = sorted(most_freq_words)
            result = SortedSet(sorted_words)
        else:
            result = self._types
        print(f'Vocabulary contains {len(result)} types')
        return result

    @cached_property
    def w2id(self) -> Dict[str, int]:
        result = {word: n for n, word in enumerate(self.types)}
        return result

    def set_tokens(self, tokens):
        """
        used if tokens should be overwritten, e.g. with a reordered list.
        this method must invalidate any caches that use old tokens
        """
        assert len(tokens) == self.num_tokens  # otherwise self.num_tokens would be incorrect
        assert set(tokens) == self.types  # otherwise self.types would be incorrect

        # overwrite
        self.__dict__['tokens'] = tokens

        # invalidate cache
        if 'token_ids' in self.__dict__:
            del self.__dict__['token_ids']

    @cached_property
    def tokens(self) -> List[str]:
        result = []
        for token in self.tokens_no_oov:
            if token in self.w2id:
                result.append(token)
            else:
                result.append(self.oov)
        return result

    @cached_property
    def token_ids(self) -> List[int]:
        result = [self.w2id[token] for token in self.tokens]
        return result

    @cached_property
    def oov_id(self) -> int:
        result = self.w2id[self.oov]
        return result

    @cached_property
    def num_tokens(self) -> int:
        result = len(self.tokens)
        return result

    @cached_property
    def w2f(self) -> Dict[str, int]:
        result = Counter(self.tokens)
        return result