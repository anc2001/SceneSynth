# Represents a program to be executed by the compiler
# Defines the syntax of the language and the semantics of the program
# Clases Program, Constraint 
# Program contains a room, query object, list of constraints (program lines) that describe all the possible placements of the object in the room
# Constraint is a tuple of (constraint_type, constraint_args)

# Need conversion from program structure tree -> program sequence token list, and the converse 

import numpy as np
from scene import Scene
from object import Furniture
from main.compiler import compile

class Node():
    # self.type -> or, and, leaf
    # self.left -> left node
    # self.right -> right node
    # self.mask -> mask at the current node
    # self.constraint -> only applicable if leaf node, 
    def __init__(self, type, constraint = None) -> None:
        if type == 'or' or type == 'and' or type == 'leaf':
            self.type = type
        else:
            print(f"invalid node type:{type}")
            exit()
        
        if type == 'leaf' and constraint:
            self.constraint = constraint
        else:
            print("Attempted to create leaf node without constraint")
            exit()

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
    # self.scene -> scene 
    # self.object -> query object to place in scene 
    # self.root -> root node of tree 
    def __init__(self, constraint = None) -> None:
        self.root = Node('leaf')

    def from_tokens(self, tokens : dict) -> None:
        structure_sequence = tokens['structure']
        constraint_sequence = tokens['constraints']

    def to_tokens(self) -> dict:
        tokens = dict()

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
        pass

