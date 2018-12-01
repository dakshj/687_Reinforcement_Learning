import itertools

import numpy as np


def random_hyperparameter_search(*param_lists) -> tuple:
    """
    Accepts multiple lists of hyperparameters, and returns random tuples
    (with repetition).

    :param param_lists: Any number of lists of hyperparameters
    """
    product = list(itertools.product(*np.array(param_lists)))
    np.random.shuffle(product)

    for params_tuple in product:
        yield params_tuple


# Example Usage
if __name__ == '__main__':
    for params in random_hyperparameter_search(
            [1, 2, 3],
            [True, False],
            ['a', 'b', 'c'],
    ):
        print(params)
