from collections import Counter, OrderedDict
from cached_property import cached_property
from sortedcontainers import SortedSet
from itertools import islice
from typing import List

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
                 num_types: int,
                 oov: str = config.Symbols.OOV,
                 _types: list = None,  # pass a vocabulary when tokens originate in test split
                 ):

        self.num_parts = num_parts
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_types = num_types
        self.oov = oov

        self._types = _types

        self.tokens_no_oov = self.prune(tokens)  # now tokens does not have to be stored in memory
        del tokens

    def make_pruning_length(self, num_raw, max_num_docs=2048):
        """
        Find length by which to prune corpora such that result is divisible by num_docs and
        such that the result of this division must be divisible by batch_size
        after first subtracting num_words_in_window.
        One cannot simply add num_words_in_window*num_docs because this might give result that is
        larger than number of available corpora.
        One can't use num_words_in_window to calculate the factor, because num_words_in_window
        should only be added once only to each document
        """
        # factor
        num_words_in_window = self.context_size + 1
        factor = self.batch_size * (max_num_docs if self._types is None else 1) + num_words_in_window
        # make divisible
        num_factors = num_raw // factor
        result = num_factors * factor - ((num_factors - (self.num_parts if self._types is None else 1))
                                         * num_words_in_window)
        return result

    def prune(self, raw):
        num_raw = len(raw)
        pruning_length = self.make_pruning_length(num_raw)
        pruned = raw[:pruning_length]
        print('Pruned {:,} total corpora to {:,}'.format(num_raw, pruning_length))
        return pruned

    # /////////////////////////////////////////////////// properties

    @cached_property
    def w2f_no_oov(self):
        c = Counter(self.tokens_no_oov)
        result = OrderedDict(
            sorted(c.items(), key=lambda item: (item[1], item[0]), reverse=True))  # order matters
        return result

    @cached_property
    def types(self):
        if self._types is None:
            most_freq_words = list(islice(self.w2f_no_oov.keys(), 0, self.num_types))
            sorted_words = sorted(most_freq_words[:-1] + [self.oov])
            result = SortedSet(sorted_words)
        else:
            result = self._types
        return result

    @cached_property
    def w2id(self):
        result = {word: n for n, word in enumerate(self.types)}
        return result

    @cached_property
    def tokens(self):
        result = []
        for token in self.tokens_no_oov:
            if token in self.w2id:
                result.append(token)
            else:
                result.append(self.oov)
        return result

    @cached_property
    def token_ids(self):
        result = [self.w2id[token] for token in self.tokens]
        return result

    @cached_property
    def oov_id(self):
        result = self.w2id[self.oov]
        return result

    @cached_property
    def num_tokens(self):
        result = len(self.tokens)
        return result

    @cached_property
    def w2f(self):
        result = Counter(self.tokens)
        return result