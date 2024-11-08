import numpy as np

idx_test = [0, 1, 2, 3, 4]
pool = [0, 5, 6, 7, 8, 9]
all_test_idx = list(set(idx_test).union(set(pool)))
print(all_test_idx)

test_idx_in_test = list(set(idx_test)-set(pool))
print(test_idx_in_test)

import numpy as np