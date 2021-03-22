"""
This Prep is an improved version of PartitionedPrep
because:
- it can generate batches of windows that slide incrementally over data
- it uses B-BPE tokenization, optionally
- handles both train and test tokens in one object
- prunes tokens more conservatively

"""

from typing import List, Optional, Generator, Tuple, Dict
import numpy as np
import random
from itertools import chain, repeat
import itertools
from numpy.lib.stride_tricks import as_strided
import string
from sortedcontainers import SortedSet
from cached_property import cached_property

from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer, AddedToken


class FlexiblePrep:
    """
    generate batches containing windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    chose to slide over tokens using sliding batch, or iterate over partitions of tokens.
    """

    def __init__(self,
                 sentences: List[str],
                 reverse: bool,
                 sliding: bool,
                 num_types: Optional[int] = None,  # can be None
                 num_parts: int = 2,
                 num_iterations: Tuple[int, int] = (8, 32),
                 batch_size: int = 64,
                 context_size: int = 7,
                 shuffle_within_part: bool = False,
                 shuffle_sentences: bool = False,
                 special_tokens: Optional[List[str]] = None,
                 min_num_test_tokens: int = 0,
                 ):

        if not isinstance(sentences, list):
            raise TypeError('Input to Prep must be a list of sentences of type List[str]')

        # check for non-ascii characters and new-lines
        for s in sentences:
            if '\n' in s:
                raise ValueError('Remove all newline characters before passing text to Prep')

            for char in set(s):
                if char != ' ' and char not in set(string.ascii_lowercase + string.punctuation + string.digits):
                    raise ValueError(f'Character "{char}" not allowed in Prep')

        if special_tokens is None:
            special_tokens = []

        self.reverse = reverse
        self.sliding = sliding
        self._num_types = num_types
        self.num_parts = num_parts
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.context_size = context_size
        self.shuffle_within_part = shuffle_within_part
        self.shuffle_sentences = shuffle_sentences
        self.special_tokens = special_tokens
        self.min_num_test_tokens = min_num_test_tokens

        # shuffle at sentence level
        # # note: this removes clustering of same-age sentences within parts, necessary when training in shuffled order
        if self.shuffle_sentences:
            random.shuffle(sentences)

        # tokenize text
        text = ' '.join(sentences)
        if num_types is not None:
            for special_token in special_tokens:
                assert special_token in text
            # only use tokenizer for tokenization, but not vocab
            print(f'Train B-BPE tokenizer with vocab size={num_types:,}')
            tokenizer = ByteLevelBPETokenizer(lowercase=True)
            tokenizer.train_from_iterator(sentences,
                                          vocab_size=num_types,
                                          min_frequency=1,
                                          # must set single_word=True
                                          special_tokens=[AddedToken(t, single_word=True) for t in special_tokens],
                                          )
            print('Tokenizing text with Byte-Level BPE...')
            tokens = [t.lstrip('Ġ').strip() for t in tokenizer.encode(text,
                                                                      add_special_tokens=True,
                                                                      ).tokens]

        else:
            print('Tokenizing text by splitting on white space...')
            tokens = text.split()

        # remove empty tokens
        tokens = [t for t in tokens if t not in {'Ġ', '', ' '}]
        print(f'{len(text.split()):,}|{len(tokens):,} tokens before|after tokenization. ')
        print(f'Encoded text with {len(set(tokens)):,}types.')

        # check that added tokens were not split during tokenization
        for special_t in special_tokens:
            if special_t not in tokens and special_t in text.split():
                print(f'"{special_t:<24}" occurs {text.split().count(special_t)} times in raw text but not in tokenized text.')

        # find num_tokens_train so that batching works
        # implementation note: find largest number of batches,
        # that can 1) fit into a part, and 2) leave enough tokens for testing
        num_tokens_train = self.calc_num_tokens(tokens, self.num_parts, min_num_test_tokens)
        # use remaining tokens as validation tokens
        tokens_pruned = tokens[num_tokens_train:]
        num_tokens_valid = self.calc_num_tokens(tokens_pruned, 1, 0)

        # split into train/valid
        self.tokens_train = tokens[:num_tokens_train]  # TODO remove tokens from middle, not from end
        self.tokens_valid = tokens_pruned[:num_tokens_valid]
        print(f'Num tokens in train={len(self.tokens_train):,}')
        print(f'Num tokens in valid={len(self.tokens_valid):,}')

    def calc_num_tokens(self,
                        _tokens,
                        num_parts: int,  # depends on train vs. test
                        min_num_remaining_tokens,
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
        return [t for t in SortedSet(self.tokens)]

    @cached_property
    def num_types(self):
        return len(self.types)

    @cached_property
    def num_tokens(self):
        return len(self.tokens)

    @cached_property
    def token2id(self) -> Dict[str, int]:
        if self.num_types is None:
            raise AttributeError('num_types must not be None to build vocabulary')
        return {t: n for n, t in enumerate(self.types)}

    # /////////////////////////////////////////////////////////////////// basic properties

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
    def num_tokens_in_part(self) -> int:
        result = self.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def midpoint(self) -> int:
        res = self.num_tokens // 2
        assert res * 2 == self.num_tokens
        return res

    # /////////////////////////////////////////////////////////////////// mbs

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
                         shape=(num_parts, self.num_tokens_in_part),
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
                         is_test: bool = False,
                         iterate_once: bool = False,
                         ) -> Generator[np.ndarray, None, None]:
        """
        yield from 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size).
        a window is an array of token IDs for context words, a target word and the next-word.
        """

        if is_test:
            if self.min_num_test_tokens == 0:
                raise ValueError('Cannot generate batches when is_test=True and min_num_test_tokens=0')

        if is_test:
            num_parts = 1
        else:
            num_parts = self.num_parts

        if iterate_once or is_test:  # useful for computing perplexity on train split
            num_iterations_list = [1] * num_parts
        else:
            num_iterations_list = self.num_iterations_list

        reordered_parts = self.get_reordered_parts(self.tokens, num_parts)

        for part_id, part in enumerate(reordered_parts):
            num_iterations = num_iterations_list[part_id]

            batches = self.batches_from_strides(part, num_iterations)

            # get batches by sliding incrementally across tokens
            if self.sliding:
                yield from batches
            # get batches by iterating over ordered partitions of tokens
            else:
                yield from chain.from_iterable(repeat(tuple(batches), num_iterations))
