from main.common.object import BBox, LineSeg

"""
tokens to tree explanation 

Example: 
structure_sequence = np.array(['and', 'or', 'c', 'c', 'c'])
constraints = [[0, 0, 1, 0], [0, 0, 1, 2], [3, 0, 1, 4]]

constraint_reference_key (structure sequence idx -> constraint sequence idx) 
= {2 : 0, 3 : 1, 4 : 2}

'and'  <-

structure: 'or', 'c', 'c', 'c'
index_tracker = [1, 2, 3, 4]

    'and'
    /
  'or' <-
structure: 'c', 'c', 'c'
index_tracker = [2, 3, 4]

Evaluate the 'c' token using the constraint_reference_key (2 -> 0), constraint at index 0

    'and'
    /
  'or'
  /
0 <-
structure: 'c', 'c'
index_tracker = [3, 4]

return to the or node and generate tree based on the remaining sequence starting on right 
    'and'
    /
 'or'  
 /  \
0    1 <-
structure: 'c'
index_tracker = [4]

Return to root 'and' node with this sequence 
    'and'
    /   \ 
 'or'    2 
 /  \
0    1 <-
"""