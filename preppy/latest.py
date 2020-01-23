"""
a new and improved Prep class based on numpy.lib.stride_tricks.
like the legacy implementation, the new class prepares ordered language data,
but does not use iterations or partitions to do so.
instead, batches of windows are created by:
 - removing the last window of the previous batch, and
 - inserting a new window in its place
this has the effect of presenting data in a more "incremental" fashion to an RNN.
"""

from typing import List, Generator, Union
from cached_property import cached_property
import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import reduce
from operator import iconcat

from preppy.tokenstore import TokenStore


class Prep:
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
                 num_evaluations: int,
                 vocab: Union[List[str], None] = None,
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
        result = self.num_windows // self.slide_size  # TODO is this correct?
        return int(result)

    # /////////////////////////////////////////////////////////////////// mbs

    @cached_property
    def last_mb(self) -> int:
        return self.num_mbs

    @cached_property
    def eval_mbs(self) -> List[int]:
        """
        self.stop_mb % self.num_evaluations == 0 is not guaranteed to be True, but that is okay
        """
        mbs_in_timepoint = self.last_mb // self.num_evaluations
        end = mbs_in_timepoint * self.num_evaluations + mbs_in_timepoint
        eval_mbs = list(range(0, end, mbs_in_timepoint))
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
    def reordered_windows(self) -> np.ndarray:
        """
        not used during training, but is useful for offline analysis of data
        """

        if self.reverse:
            token_ids_array = np.array(self.store.token_ids[::-1], dtype=np.int16)
        else:
            token_ids_array = np.array(self.store.token_ids, dtype=np.int16)

        if token_ids_array.dtype == np.int16:
            stride = 2  # bytes because 16 bits = 2 bytes
        else:
            raise ValueError('Careful: Stride must be changed if data-type is changed')

        num_possible_windows = len(token_ids_array) - self.num_tokens_in_window
        shape = (num_possible_windows, self.num_tokens_in_window)
        all_windows = as_strided(token_ids_array, shape, strides=(stride, stride), writeable=False)
        print(f'Matrix containing all windows has shape={all_windows.shape}')

        return all_windows

    def gen_windows(self) -> Generator[np.ndarray, None, None]:

        if self.reverse:
            token_ids_array = np.array(self.store.token_ids[::-1], dtype=np.int16)
        else:
            token_ids_array = np.array(self.store.token_ids, dtype=np.int16)

        if token_ids_array.dtype == np.int16:
            stride = 2  # bytes because 16 bits = 2 bytes
        else:
            raise ValueError('Careful: Stride must be changed if data-type is changed')

        # generate batches of windows - implementations is memory efficient because as_strided() returns views
        for windows in as_strided(token_ids_array,
                                  shape=(self.num_mbs, self.batch_size, self.num_tokens_in_window),
                                  strides=(stride * self.slide_size, stride, stride),
                                  writeable=False):
            yield windows
