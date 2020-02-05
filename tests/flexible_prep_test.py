import unittest
import numpy as np
from pathlib import Path

from preppy.flexible import FlexiblePrep


class MyTest(unittest.TestCase):

    def test_gen_windows(self):
        """
        test that Preppy legacy.TrainPrep.gen_windows() works as expected
        """

        # load docs
        corpus_path = Path(__file__).parent / 'test_docs.txt'
        text_in_file = corpus_path.read_text()
        docs = text_in_file.split('\n')

        # prep
        reverse = False
        num_iterations = (2, 1)
        context_size = 2
        num_parts = 2
        batch_size = 6

        prep_s = FlexiblePrep(docs,
                              reverse=reverse,
                              sliding=True,
                              num_iterations=num_iterations,
                              context_size=context_size,
                              num_parts=num_parts,
                              batch_size=batch_size,
                              )

        prep_p = FlexiblePrep(docs,
                              reverse=reverse,
                              sliding=False,
                              num_iterations=num_iterations,
                              context_size=context_size,
                              num_parts=num_parts,
                              batch_size=batch_size,
                              )

        # test
        for n, bs in enumerate(prep_s.generate_batches()):
            print(n + 1)
            print(bs)

            # check that integers are valid token IDs
            is_outside_range = np.any((bs < 0) | (bs > len(prep_s.store.types)))
            self.assertFalse(is_outside_range)

        len_s = len(list(prep_s.generate_batches()))
        len_p = len(list(prep_p.generate_batches()))
        self.assertTrue(len_s == len_p)


if __name__ == '__main__':
    unittest.main()