from collections import Counter, OrderedDict
from cached_property import cached_property
from sortedcontainers import SortedSet
from itertools import islice
import random
from itertools import chain

from preppy import config


def make_terms(params):
    def load_lines(item_name):
        f = config.Dirs.items / '{}_{}.txt'.format(params.corpus_name, item_name)
        res = f.open('r').readlines()
        return res

    def tokenize(all_lines, ids):
        test_lines = []
        for test_line_id in ids:
            test_line = all_lines.pop(test_line_id)  # removes line and returns removed line
            test_lines.append(test_line)

        # shuffle documents before splitting
        if params.shuffle_docs:
            # seed=3 results in equal distribution of probe words,
            # but gives better results when inc_age vs dec_age which suggest seed is bad
            # TODO test seed=3 again, because I might have plotted results incorrectly
            random.seed(None)
            random.shuffle(all_lines)
        # tokenize
        res = (list(chain(*[line.split() for line in all_lines])),
               list(chain(*[line.split() for line in test_lines])))
        return res

    # split train test
    train_raws = {}
    test_raws = {}
    for name in ['terms', 'tags']:
        lines = load_lines(name)
        print('Found {} documents'.format(len(lines)))
        num_test_line_ids = len(lines) - config.Terms.NUM_TEST_LINES  # adjust because of popping
        random.seed(3)
        test_line_ids = random.sample(range(num_test_line_ids), config.Terms.NUM_TEST_LINES)
        train_raw, test_raw = tokenize(lines, test_line_ids)
        # oov
        train_set = set(train_raw)
        for n, item in enumerate(test_raw):
            if item not in train_set:
                test_raw[n] = config.Terms.OOV_SYMBOL
        train_raws[name] = train_raw
        test_raws[name] = test_raw
    # terms
    train_terms = TermStore(train_raws['terms'], train_raws['tags'], params)
    test_terms = TermStore(test_raws['terms'], test_raws['tags'], params, types=train_terms.types)
    result = train_terms, test_terms
    return result


class TermStore(object):
    """
    Stores terms.
    """

    def __init__(self, terms, tags, params, types=None):
        self.params = params
        self._types = types  # test corpora must have train types otherwise ids don't match
        self.tokens_no_oov, self.tags_no_oov = self.preprocess(terms, tags)

    # /////////////////////////////////////////////////// corpora

    def preprocess(self, terms, tags):
        raw_items = list(zip(terms, tags))
        if self._types is None:
            items = self.prune(raw_items)  # train preprocessing might be a bit different
        else:
            items = self.prune(raw_items)  # test
        result = list(zip(*items))
        return result

    def make_item_length(self, num_raw):
        """
        Find length by which to prune corpora such that result is divisible by num_docs and
        such that the result of this division must be divisible by batch_size
        after first subtracting num_items_in_window.
        One cannot simply add num_items_in_window*num_docs because this might give result that is
        larger than number of available corpora.
        One can't use num_items_in_window to calculate the factor, because num_items_in_window
        should only be added once only to each document
        """
        # factor
        num_items_in_window = self.params.window_size + 1
        factor = self.params.batch_size * (config.Terms.MAX_NUM_DOCS if self._types is None else 1) + num_items_in_window
        # make divisible
        num_factors = num_raw // factor
        result = num_factors * factor - ((num_factors - (self.params.num_parts if self._types is None else 1))
                                         * num_items_in_window)
        return result

    def prune(self, raw):
        num_raw = len(raw)
        item_length = self.make_item_length(num_raw)
        pruned = raw[:item_length]
        print('Pruned {:,} total corpora to {:,}'.format(num_raw, item_length))
        return pruned

    # /////////////////////////////////////////////////// terms

    @cached_property
    def type_freq_dict_no_oov(self):
        c = Counter(self.tokens_no_oov)
        result = OrderedDict(
            sorted(c.items(), key=lambda item: (item[1], item[0]), reverse=True))  # order matters
        return result

    @cached_property
    def types(self):
        if self._types is None:
            most_freq_items = list(islice(self.type_freq_dict_no_oov.keys(), 0, self.params.num_types))
            sorted_items = sorted(most_freq_items[:-1] + [config.Terms.OOV_SYMBOL])
            result = SortedSet(sorted_items)
        else:
            result = self._types
        return result

    @cached_property
    def term_id_dict(self):
        result = {item: n for n, item in enumerate(self.types)}
        return result

    @cached_property
    def tokens(self):
        result = []
        for token in self.tokens_no_oov:
            if token in self.term_id_dict:
                result.append(token)
            else:
                result.append(config.Terms.OOV_SYMBOL)
        return result

    @cached_property
    def token_ids(self):
        result = [self.term_id_dict[token] for token in self.tokens]
        return result

    @cached_property
    def oov_id(self):
        result = self.term_id_dict[config.Terms.OOV_SYMBOL]
        return result

    @cached_property
    def num_tokens(self):
        result = len(self.tokens)
        return result

    @cached_property
    def term_freq_dict(self):
        result = Counter(self.tokens)
        return result