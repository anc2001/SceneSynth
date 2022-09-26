from main import common
from main.common.language import ProgramTree

import numpy as np

structure_sequence = np.array(['or', 'and', 'or', 'c', 'c', 'c', 'or', 'c', 'c'])
constraints = [[0, 0, 1, 0], [0, 0, 1, 2], [3, 0, 1, 4], [0, 0, 1, 0], [0, 0, 1, 2]]

# structure_sequence = np.array(['and', 'or', 'c', 'c', 'c'])
# constraints = [[0, 0, 1, 0], [0, 0, 1, 2], [3, 0, 1, 4]]

# structure_sequence = np.array(['or', 'c', 'c'])
# constraints = [[0, 0, 1, 0], [0, 0, 1, 2]]

# structure_sequence = np.array(['c'])
# constraints = [[0, 0, 1, 2]]

# structure_sequence = np.array(['or', 'and', 'or', 'c', 'c', 'c', 'or', 'c', 'c', 'c'])
# constraints = [[0, 0, 1, 0], [0, 0, 1, 2], [3, 0, 1, 4], [0, 0, 1, 0], [0, 0, 1, 2]]

# structure_sequence = np.array(['or', 'c', 'c', 'c', 'and'])
# constraints = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]]
tokens = {
    'structure' : structure_sequence,
    'constraints' : constraints
}

tree = ProgramTree()
tree.from_tokens(tokens)
tree.print_program()

test_tokens = tree.to_tokens()
print(test_tokens['structure'])
print(test_tokens['constraints'])