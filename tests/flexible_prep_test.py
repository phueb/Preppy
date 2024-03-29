import unittest
import numpy as np


from preppy.flexible import Prep


class MyTest(unittest.TestCase):

    def test_gen_windows(self):
        """
        test that FlexiblePrep.gen_windows() works as expected,
        by comparing out put of sliding=False to sliding=True
        """

        sentences = [
            'd1w01 d1w02 d1w03 d1w04 d1w05 d1w06 d1w07 d1w08 d1w09 d1w10 d1w11 d1w12 d1w13 d1w14 d1w15 d1w16 d1w17',
            'd2w01 d2w02 d2w03 d2w04 d2w05 d2w06 d2w07 d2w08 d2w09 d2w10 d2w11 d2w12 d2w13 d2w14 d2w15 d2w16 d2w17',
            'd3w01 d3w02 d3w03 d3w04 d3w05 d3w06 d3w07 d3w08 d3w09 d3w10 d3w11 d3w12 d3w13 d3w14',
            'd4w01 d4w02 d4w03 d4w04 d4w05 d4w06 d4w07 d4w08 d4w09 d4w10 d4w11 d4w12 d4w13 d4w14',
        ]

        # prep
        reverse = False
        num_iterations = (2, 1)
        context_size = 2
        num_parts = 2
        batch_size = 6

        prep_s = Prep(sentences,
                      reverse=reverse,
                      sliding=True,
                      num_iterations=num_iterations,
                      context_size=context_size,
                      num_parts=num_parts,
                      batch_size=batch_size,
                      )

        prep_p = Prep(sentences,
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
            is_outside_range = np.any((bs < 0) | (bs > len(prep_s.types)))
            self.assertFalse(is_outside_range)

        len_s = len(list(prep_s.generate_batches()))
        len_p = len(list(prep_p.generate_batches()))
        self.assertTrue(len_s == len_p)


if __name__ == '__main__':
    unittest.main()