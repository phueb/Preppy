from collections import Counter, OrderedDict
from cached_property import cached_property
from sortedcontainers import SortedSet
from itertools import islice
from typing import List, Dict, Union, Optional

from preppy import config


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
                 oov: str = config.Symbols.OOV,
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
        Find length by which to prune tokens such that result is divisible by num_docs and
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