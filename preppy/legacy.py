"""
includes logic for handling multiple iterations.
multiple iterations were used in 2019 master's thesis of PH.

"""

from typing import List
from cached_property import cached_property
import numpy as np


class Prep:
    def __init__(self,
                 docs: List[str],
                 num_parts: int,
                 part_order: str,
                 num_types: int,
                 num_iterations: List[int],
                 batch_size: int,
                 window_size: int,
                 num_evaluations: int
                 ):

        self.docs = docs
        self.num_parts = num_parts
        self.part_order = part_order
        self.num_types = num_types
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_evaluations = num_evaluations

        print('Initialized Preppy')

    @cached_property
    def num_mbs_in_part(self):
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_test(self):
        result = self.num_windows_in_test / self.batch_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_iterations_list(self):
        result = np.linspace(self.num_iterations[0], self.num_iterations[1],
                             num=self.num_parts, dtype=np.int)
        return result

    @cached_property
    def mean_num_iterations(self):
        result = np.mean(self.num_iterations_list)
        return result

    @cached_property
    def num_mbs_in_block(self):
        result = self.num_mbs_in_part * self.mean_num_iterations
        return result

    @cached_property
    def num_mbs_in_token_ids(self):
        result = self.num_mbs_in_part * self.num_parts
        return int(result)

    @cached_property
    def num_items_in_window(self):
        num_items_in_window = self.window_size + 1
        return num_items_in_window

    @cached_property
    def num_windows_in_part(self):
        result = self.num_items_in_part - self.num_items_in_window
        return result

    @cached_property
    def num_windows_in_test(self):
        result = self.test_terms.num_tokens - self.num_items_in_window  # TODO test
        return result

    @cached_property
    def num_items_in_part(self):
        result = self.train_terms.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def stop_mb(self):
        stop_mb = self.num_parts * self.num_mbs_in_block
        return stop_mb

    @cached_property
    def eval_mbs(self):
        mbs_in_timepoint = int(self.stop_mb / self.num_evaluations)
        end = mbs_in_timepoint * self.num_evaluations + mbs_in_timepoint
        eval_mbs = list(range(0, end, mbs_in_timepoint))
        return eval_mbs

    def make_windows_mat(self, part, num_windows):
        result = np.zeros((num_windows, self.num_items_in_window), dtype=np.int)
        for window_id in range(num_windows):
            window = part[window_id:window_id + self.num_items_in_window]
            result[window_id, :] = window
        return result

    def gen_ids(self, num_iterations_list=None, is_test=False):
        if not is_test:
            parts = self.reordered_partitions
            num_mbs_in_part = self.num_mbs_in_part
            num_windows = self.num_windows_in_part
        else:
            parts = [self.test_terms.token_ids]
            num_mbs_in_part = self.num_mbs_in_test
            num_windows = self.num_windows_in_test
        if not num_iterations_list:
            num_iterations_list = self.num_iterations_list
        # generate
        for part_id, part in enumerate(parts):
            windows_mat = self.make_windows_mat(part, num_windows)
            windows_mat_x, windows_mat_y = np.split(windows_mat, [self.params.window_size], axis=1)
            num_iterations = num_iterations_list[part_id]
            print('Iterating {} times over part {}'.format(num_iterations, part_id))
            for _ in range(num_iterations):
                for x, y in zip(np.vsplit(windows_mat_x, num_mbs_in_part),
                                np.vsplit(windows_mat_y, num_mbs_in_part)):
                    yield x, y