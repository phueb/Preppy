"""
legacy prep
includes logic for handling multiple iterations.
multiple iterations were used in 2019 master's thesis of PH.

"""

from typing import List, Generator
from cached_property import cached_property
import numpy as np
from functools import reduce
from operator import iconcat

from preppy.tokenstore import TokenStore


class Prep:
    """
    text is split into 2 partitions - more then 2 partitions is not supported.
    this means that input must already be ordered as desired in the text file.

    a window consists of a multi-word context + a single word (which is predicted)

    """
    def __init__(self,
                 docs: List[str],
                 reverse: bool,
                 num_types: int,
                 num_iterations: List[int],
                 batch_size: int,
                 context_size: int,
                 num_evaluations: int,
                 vocab: List[str] = None,
                 ):

        self.num_parts = 2  # this project is for reference/demonstration only so no need to change

        tokens = reduce(iconcat, docs, [])  # flattens list of lists
        self.store = TokenStore(tokens, self.num_parts, batch_size, context_size, num_types, vocab)
        self.reverse = reverse
        self.num_types = num_types
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_evaluations = num_evaluations

        print(f'Initialized Prep with {self.store.num_tokens} tokens')

    @cached_property
    def num_mbs_in_part(self) -> int:
        result = self.num_windows_in_part / self.batch_size
        assert result.is_integer()
        return int(result)

    @cached_property
    def num_iterations_list(self) -> List[int]:
        result = list(np.linspace(self.num_iterations[0], self.num_iterations[1],
                                  num=self.num_parts, dtype=np.int))
        return result

    @cached_property
    def mean_num_iterations(self) -> int:
        result = np.mean(self.num_iterations_list)
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def num_mbs_in_block(self) -> int:
        return self.num_mbs_in_part * self.mean_num_iterations

    @cached_property
    def num_mbs_in_token_ids(self) -> int:
        return self.num_mbs_in_part * self.num_parts

    @cached_property
    def num_tokens_in_window(self) -> int:
        return self.context_size + 1

    @cached_property
    def num_windows_in_part(self) -> int:
        return self.num_tokens_in_part - self.num_tokens_in_window

    @cached_property
    def num_tokens_in_part(self) -> int:
        result = self.store.num_tokens / self.num_parts
        assert float(result).is_integer()
        return int(result)

    @cached_property
    def stop_mb(self) -> int:
        return self.num_parts * self.num_mbs_in_block

    @cached_property
    def eval_mbs(self) -> List[int]:
        assert self.stop_mb % self.num_evaluations == 0
        mbs_in_timepoint = self.stop_mb // self.num_evaluations
        end = mbs_in_timepoint * self.num_evaluations + mbs_in_timepoint
        eval_mbs = list(range(0, end, mbs_in_timepoint))
        return eval_mbs

    # /////////////////////////////////////////////////////////////////// parts & windows

    @cached_property
    def midpoint(self) -> int:
        res = self.store.num_tokens // 2
        assert res * 2 == self.store.num_tokens
        return res

    @cached_property
    def part1(self) -> List[int]:
        return self.store.token_ids[:self.midpoint]

    @cached_property
    def part2(self) -> List[int]:
        return self.store.token_ids[self.midpoint:]

    @cached_property
    def reordered_parts(self) -> List[List[int]]:
        if self.reverse:
            return [self.part2, self.part1]
        else:
            return [self.part1, self.part2]

    def make_windows_mat(self,
                         part: List[int],
                         num_windows: int
                         ) -> np.ndarray:
        result = np.zeros((num_windows, self.num_tokens_in_window), dtype=np.int)
        for window_id in range(num_windows):
            window = part[window_id:window_id + self.num_tokens_in_window]
            result[window_id, :] = window
        return result

    def gen_windows(self, iterate_once=False) -> Generator[np.ndarray, None, None]:
        """
        this was previously called "gen_ids", and it generated x, y rather than windows
        """
        if not iterate_once:
            num_iterations_list = self.num_iterations_list
        else:
            num_iterations_list = [1] * self.num_parts

        # generate
        for part_id, part in enumerate(self.reordered_parts):
            windows_mat = self.make_windows_mat(part, self.num_windows_in_part)
            num_iterations = num_iterations_list[part_id]
            print('Iterating {} times over part {}'.format(num_iterations, part_id))

            for _ in range(num_iterations):
                for windows in np.vsplit(windows_mat, self.num_mbs_in_part):
                    yield windows