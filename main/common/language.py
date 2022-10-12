from main.common.scene import Scene
from main.common.object import Furniture
from main.compiler import ensure_placement_validity, solve_constraint, \
    convert_mask_to_image, \
    collapse_mask
from main.common.utils import raise_exception

from main.config import constraint_types, image_filepath, grid_size

import matplotlib.image as img
import numpy as np
import os

class Node():
    """
    self.type -> or, and, leaf
    self.left -> left node
    self.right -> right node
    self.mask -> mask at the current node
    self.constraint -> only applicable if leaf node
    """
    def __init__(self, type, constraint = None) -> None:
        self.type = type
        self.constraint = constraint
    
    def __len__(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def is_leaf(self):
        return self.type == 'leaf'

    def evaluate(self, scene : Scene, query_object : Furniture) -> np.ndarray:
        # returns a 3D array representing the binary mask of all possible object placements in the room
        name = str(len(self))
        if self.type == 'leaf':
            self.mask = solve_constraint(self.constraint, scene, query_object)
            name = name + "_" + constraint_types[self.constraint[0]]
        else:
            mask1 = self.left.evaluate(scene, query_object)
            mask2 = self.right.evaluate(scene, query_object)
            csg_operator = np.logical_and if self.type == 'and' else np.logical_or
            self.mask = csg_operator(mask1, mask2)
            name = name + "_" + self.type

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
        
        # See below for code explanation 
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
        mask_4d = self.root.evaluate(scene, query_object)
        # Collapse 4D mask into 3D mask and ensure placement validity inside the room 
        mask_3d = collapse_mask(mask_4d)
        ensure_placement_validity(mask_3d, scene, query_object)
        self.mask = mask_3d

        return self.mask

    def print_program(self, scene, query_object, parent_folder) -> None:
        fout = open(os.path.join(parent_folder, "program.txt"), "w")
        def print_program_helper(node, count):
            if node.is_leaf():
                mask_name = f"mask_{count}"
                fout.write(f"{mask_name} = {node.constraint}\n")
                image = convert_mask_to_image(node.mask, scene)
                img.imsave(
                    os.path.join(parent_folder, mask_name + '.png'), 
                    image
                )
                return mask_name, count + 1
            else:
                left_name, new_count = print_program_helper(node.left, count)
                right_name, newer_count = print_program_helper(node.right, new_count)
                mask_name = f"mask_{newer_count}"
                fout.write(f"{mask_name} = {left_name} {node.type} {right_name}\n")
                image = convert_mask_to_image(node.mask, scene)
                img.imsave(
                    os.path.join(parent_folder, mask_name + '.png'), 
                    image
                )
                return mask_name, newer_count + 1

        final_name, _ = print_program_helper(self.root, 0)
        fout.write(f"return {final_name}\n")
        fout.close()

        image = convert_mask_to_image(self.mask, scene)
        img.imsave(
            os.path.join(parent_folder, 'final.png'), 
            image
        )

        scene_mask = scene.convert_to_mask()
        scene_mask = ~scene_mask.astype(bool)
        mask_image = np.repeat(                
            np.expand_dims(
                scene_mask, 
                axis = 2
            ),
            3,
            axis = 2
        )
        mask_image = np.rot90(mask_image).astype(float)
        img.imsave(
            os.path.join(parent_folder, 'scene_mask.png'), 
            mask_image
        )

        image = np.zeros((grid_size, grid_size, 3))
        query_object.write_to_image(scene, image, normalize= True)
        image = np.rot90(image)
        img.imsave(
            os.path.join(parent_folder, 'query_object.png'), 
            image
        )

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