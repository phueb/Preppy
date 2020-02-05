"""
a new and improved Prep class based on numpy.lib.stride_tricks.
like the legacy implementation, the new class prepares ordered language data,
but does not use iterations or partitions to do so.
instead, batches of windows are created by:
 - removing the last window of the previous batch, and
 - inserting a new window in its place
this has the effect of presenting data in a more "incremental" fashion to an RNN.
"""

from typing import List, Generator, Optional
from cached_property import cached_property
import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import reduce
from operator import iconcat

from preppy.tokenstore import TokenStore


class SlidingPrep:
    """
    generates batches of windows of word IDs.
    a window consists of a multi-word context + a single word (which is predicted)

    Warning: the input must already be ordered as desired in the text file.

    this class can be used with both train and test data.
    but, when using test data, it is important to pass a pre-defined vocabulary,
    otherwise a new vocabulary will be created (and word IDS will be re-assigned).
    secondly, it is recommended to adjust slide_size when using test data,
    otherwise, for small slide_size, identical windows may be fed to model multiple times,
    causing redundant computations.

    """

    def __init__(self,
                 docs: List[str],
                 reverse: bool,
                 num_types: int,
                 slide_size: int,
                 batch_size: int,
                 context_size: int,
                 num_evaluations: int = 10,
                 vocab: Optional[List[str]] = None,  # pass a vocab when initializing with held-out documents
                 ):

        # make store
        tokenized_docs = [d.split() for d in docs]
        tokens = reduce(iconcat, tokenized_docs, [])  # flatten list of lists
        num_parts = 1
        self.store = TokenStore(tokens, num_parts, batch_size, context_size, num_types, vocab)

        self.reverse = reverse
        self.num_types = num_types
        self.slide_size = slide_size
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_evaluations = num_evaluations

        self.token_ids_array = np.array(self.store.token_ids, dtype=np.int64)
        if self.token_ids_array.dtype == np.int64:
            self.stride = 8  # bytes because 64 bits = 2 bytes ; changing this may cause CUDA error
        else:
            raise ValueError('Stride must be changed if data-type is changed')

    @cached_property
    def num_tokens(self) -> int:
        result = self.store.num_tokens
        return result

    @cached_property
    def num_tokens_in_window(self) -> int:
        return self.context_size + 1

    @cached_property
    def num_windows(self) -> int:
        return self.num_tokens - self.num_tokens_in_window

    @cached_property
    def num_mbs(self) -> int:
        result = self.num_windows // self.slide_size
        return int(result)

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

    # /////////////////////////////////////////////////////////////////// windows

    @cached_property
    def midpoint(self) -> int:
        res = self.store.num_tokens // 2
        assert res * 2 == self.store.num_tokens
        return res

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

    def gen_windows(self) -> Generator[np.ndarray, None, None]:
        """yield from 3d array where each 2d slice is a batch of windows with shape (batch_size, context_size)"""
        batches = as_strided(self.token_ids_array,
                             shape=(self.num_mbs, self.batch_size, self.num_tokens_in_window),
                             strides=(self.stride * self.slide_size, self.stride, self.stride),
                             writeable=False)
        if self.reverse:
            yield from np.flip(batches, axis=0)
        else:
            yield from batches
