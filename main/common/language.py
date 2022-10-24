from main.common.scene import Scene
from main.common.object import Furniture
from main.compiler import \
    ensure_placement_validity, solve_constraint, \
    convert_mask_to_image
from main.common.utils import raise_exception

from main.config import grid_size, \
    direction_types_map, constraint_types_map

import matplotlib.image as img
import numpy as np
import os

def verify_program(tokens, query_idx):
    structure = np.array(tokens['structure'])
    constraints = tokens['constraints']
    if np.sum(structure == 'c') != len(constraints):
        return False
    for constraint in constraints:
        type = constraint[0]
        if not query_idx == constraint[1]:
            print("Query index is invalid")
            return False
        if query_idx == constraint[2]:
            print("Reference index is invalid")
            return False
        orientation_flag = type == constraint_types_map['align']
        orientation_flag |= type == constraint_types_map['face']
        direction_pad_flag = constraint[3] == direction_types_map['<pad>']
        if orientation_flag != direction_pad_flag:
            print("Type and directions don't match")
            return False
    
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
    if len(remaining_tokens):
        print("Too many tokens predicted")
    return not (len(remaining_tokens) or not valid)

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
        mask_3d = self.root.evaluate(scene, query_object)
        ensure_placement_validity(mask_3d, scene, query_object)
        self.mask = mask_3d

        return self.mask

    def print_program(
            self, scene, query_object, parent_folder,
            writer = None, base_tag = None, display_on_tensorboard = False
        ):
        program_text = []
        def print_program_helper(node, count):
            if node.is_leaf():
                mask_name = f"mask_{count}"
                program_text.append(f"{mask_name} = {node.constraint}\n")
                image = convert_mask_to_image(node.mask, scene)
                filepath = os.path.join(parent_folder, mask_name + '.png')
                img.imsave(filepath, image)
                
                if display_on_tensorboard:
                    tag = base_tag + mask_name
                    writer.add_image(tag, image, dataformats='HWC')
                return mask_name, count + 1
            else:
                left_name, new_count = print_program_helper(node.left, count)
                right_name, newer_count = print_program_helper(node.right, new_count)
                mask_name = f"mask_{newer_count}"
                program_text.append(f"{mask_name} = {left_name} {node.type} {right_name}\n")
                image = convert_mask_to_image(node.mask, scene)
                filepath = os.path.join(parent_folder, mask_name + '.png')
                img.imsave(filepath, image)
                
                if display_on_tensorboard:
                    tag = base_tag + mask_name
                    writer.add_image(tag, image, dataformats='HWC')
                return mask_name, newer_count + 1

        final_name, _ = print_program_helper(self.root, 0)
        program_text.append(f"return {final_name}\n")
        program_string = ' '.join(program_text)

        filepath = os.path.join(parent_folder, "program.txt")
        fout = open(filepath, "w")
        fout.write(program_string)
        fout.close()

        if display_on_tensorboard:
            tag = base_tag + "program"
            writer.add_text(filepath, program_string)

        image = convert_mask_to_image(self.mask, scene)
        filepath = os.path.join(parent_folder, 'final.png')
        img.imsave(filepath, image)

        if display_on_tensorboard:
            tag = base_tag + "final"
            writer.add_image(tag, image, dataformats='HWC')

        image = np.zeros((grid_size, grid_size, 3))
        query_object.write_to_image(scene, image, normalize= True)
        image = np.rot90(image)
        filepath = os.path.join(parent_folder, 'query_object.png')
        img.imsave(filepath, image)

        if display_on_tensorboard:
            tag = base_tag + "query_object"
            writer.add_image(tag, image, dataformats='HWC')