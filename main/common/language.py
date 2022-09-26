import numpy as np
from main.common.scene import Scene
from main.common.object import Furniture
from main.compiler import compile
from main.common.utils import raise_exception

class Node():
    # self.type -> or, and, leaf
    # self.left -> left node
    # self.right -> right node
    # self.mask -> mask at the current node
    # self.constraint -> only applicable if leaf node, 
    def __init__(self, type, constraint = None) -> None:
        self.type = type
        self.constraint = constraint

    def is_leaf(self):
        return self.type == 'leaf'

    def evaluate(self, scene : Scene, query_object : Furniture) -> np.ndarray:
        # returns a 3D array representing the binary mask of all possible object placements in the room
        if self.type == 'leaf':
            compile(self.constraint, scene, query_object)
        else:
            mask1 = self.left.evaluate(scene, query_object)
            mask2 = self.right.evaluate(scene, query_object)
            csg_operator = np.logical_and if self.type == 'and' else np.logical_or
            self.mask = csg_operator(mask1, mask2)
        return self.mask

class ProgramTree():
    # self.root -> root node of tree 
    
    def from_constraint(self, constraint) -> None:
        self.root = Node('leaf', constraint)

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
                raise_exception('tree')
            
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
                raise_exception('tree')

        root_node, remaining_structure, _ = parse(structure, index_tracker)
        if len(remaining_structure) > 0:
            raise_exception('tree')
        
        self.root = root_node

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

    # Convention is the self goes on the left, other on the right
    def combine(self, type, other_tree):
        new_root = Node(type)
        new_root.left = self.root
        new_root.right = other_tree
        self.root = new_root

    def evaluate(self, scene : Scene, query_object : Furniture) -> np.ndarray:
        # returns a 3D mask that can be used for evaluation 
        mask_4d = self.root.evaluate(scene, query_object)
        mask_3d = None
        return mask_3d

    def print_program(self) -> None:
        def print_program_helper(node, count):
            if node.is_leaf():
                mask_name = f"mask_{count}"
                print(f"{mask_name} = {node.constraint}")
                return mask_name, count + 1
            else:
                left_name, new_count = print_program_helper(node.left, count)
                right_name, newer_count = print_program_helper(node.right, new_count)
                mask_name = f"mask_{newer_count}"
                print(f"{mask_name} = {left_name} {node.type} {right_name}")
                return mask_name, newer_count + 1
        
        final_name, _ = print_program_helper(self.root, 0)
        print(f"return {final_name}")