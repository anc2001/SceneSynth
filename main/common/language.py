from main.common.scene import Scene
from main.common.object import Furniture
from main.executor import \
    ensure_placement_validity, solve_constraint
from main.common.tree_viz import visualize_program

from main.config import \
    direction_types_map, constraint_types_map

import numpy as np

def verify_program(tokens, query_idx):
    structure = np.array(tokens['structure'])
    constraints = tokens['constraints']
    if np.sum(structure == 'c') != len(constraints):
        error_statement = "Num of constraints predicted and needed not the same!"
        error_statement += f"Structure: {structure} \n"
        error_statement += f"Constrains: {constraints} \n"
        return False, error_statement
    for constraint in constraints:
        type = constraint[0]
        if not query_idx == constraint[1]:
            error_statement = f"Query index is invalid: {constraint}"
            return False, error_statement
        if query_idx == constraint[2]:
            error_statement = f"Reference index is invalid: {constraint}"
            return False, error_statement
        # orientation_flag = type == constraint_types_map['align']
        # orientation_flag |= type == constraint_types_map['face']
        # direction_pad_flag = constraint[3] == direction_types_map['null']
        # if orientation_flag != direction_pad_flag:
        #     error_statement = f"Type and directions don't match: {constraint}"
        #     return False, error_statement
    
    def verify_tree(sequence):
        if not len(sequence):
            return np.array([]), False 
        if sequence[0] == 'c':
            return sequence[1:], True
        elif sequence[0] == 'or' or sequence[0] == 'and':
            left_partial, validity_check = verify_tree(sequence[1:])
            right_partial, validity_check = verify_tree(left_partial)
            return right_partial, validity_check and validity_check
        else:
            return np.array([]), False
    
    remaining_tokens, valid = verify_tree(structure)
    error_statement = "Program is valid"
    if len(remaining_tokens):
        error_statement = f"Too many tokens predicted {structure}"
    validity = not (len(remaining_tokens) or not valid)
    return validity, error_statement

class Node():
    """
    self.type -> or, and, leaf
    self.left -> left node
    self.right -> right node
    self.mask -> mask at the current node
    self.constraint -> only applicable if leaf node
    """
    def __init__(self, type, constraint = []) -> None:
        self.type = type
        self.constraint = [int(val) for val in constraint]
    
    def __len__(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def is_leaf(self):
        return self.type == 'leaf'

    def evaluate(self, scene : Scene, query_object : Furniture) -> np.ndarray:
        # returns a 3D array representing the binary mask of all possible object placements in the room
        if self.type == 'leaf':
            self.mask = solve_constraint(self.constraint, scene, query_object)
        else:
            mask1 = self.left.evaluate(scene, query_object)
            mask2 = self.right.evaluate(scene, query_object)
            csg_operator = np.logical_and if self.type == 'and' else np.logical_or
            self.mask = csg_operator(mask1, mask2)

        return self.mask

class ProgramTree():
    """
    self.root -> root node of tree 
    """
    def __init__(self) -> None:
        self.root = np.array([])
        self.program_length = 0

    def __len__(self):
        return self.program_length
    
    def from_constraint(self, constraint) -> None:
        self.root = Node('leaf', constraint)
        self.program_length = 1

    def from_tokens(self, tokens : dict) -> None:
        structure = np.array(tokens['structure'])
        constraints = np.array(tokens['constraints'])
        index_tracker = np.arange(len(structure))
        # (structure sequence idx -> constraint sequence idx)
        constraint_reference_key = {
            item : i for i, item in enumerate(index_tracker[structure == 'c'])
        }
        
        def parse(tree_structure, index_tracker):
            if len(tree_structure) == 0:
                raise Exception('tree structure incorrect')
            
            if tree_structure[0] == 'c':
                constraint = constraints[
                    constraint_reference_key[
                        index_tracker[0]
                    ]
                ]
                return Node('leaf', constraint), tree_structure[1:], index_tracker[1:]
            elif tree_structure[0] == 'or' or tree_structure[0] == 'and':
                node = Node(tree_structure[0])
                left_node, right_tree_structure, right_index_tracker = parse(
                    tree_structure[1:], index_tracker[1:]
                )
                node.left = left_node
                right_node, remaining_tree_structure, remaining_index_tracker = parse(
                    right_tree_structure, right_index_tracker
                )
                node.right = right_node
                return node, remaining_tree_structure, remaining_index_tracker
            else:
                raise Exception('tree structure incorrect')

        root_node, remaining_structure, _ = parse(structure, index_tracker)
        if len(remaining_structure) > 0:
            raise Exception('tree structure incorrect')
        
        self.root = root_node
        self.program_length = len(self.root)

    def to_tokens(self) -> dict:
        def flatten(node):
            if node.is_leaf():
                return np.array(['c']), np.array([node.constraint])
            else:
                left_structure, left_constraints = flatten(node.left)
                right_structure, right_constraints = flatten(node.right)
                tree_structure = np.concatenate([[node.type], left_structure, right_structure])
                constraints = np.concatenate([left_constraints, right_constraints], axis = 0)
                return tree_structure, constraints
        structure_sequence, constraint_sequence = flatten(self.root)
        return {
            'structure' : structure_sequence,
            'constraints' : constraint_sequence
        }

    def combine(self, type, other_tree):
        # Convention is the self goes on the left, other on the right
        if type == 'or' or type == 'and':
            if len(self.root):
                new_root = Node(type)
                new_root.left = self.root
                new_root.right = other_tree.root
                self.root = new_root
                self.program_length = len(new_root)
            else:
                self.root = other_tree.root
                self.program_length = other_tree.program_length
        else:
            print("Invalid combination node type")
            return None

    def evaluate(self, scene : Scene, query_object : Furniture) -> np.ndarray:
        # returns a 3D mask that can be used for evaluation 
        mask = self.root.evaluate(scene, query_object)
        # final_mask = np.array(mask_3d)
        # ensure_placement_validity(final_mask, scene, query_object)
        # self.mask = final_mask
        self.mask = mask

        return self.mask

    def print_program(self, scene, query_object):
        # return matplotlib figure of program diagram
        return visualize_program(self, scene, query_object)