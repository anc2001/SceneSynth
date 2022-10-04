from main.common.language import ProgramTree
from main.config import \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types_map, constraint_types_map

import numpy as np

def generate_most_restrictive_program(room, query_object):
    query_object_idx = len(room.objects)
    query_semantic_fronts = query_object.world_semantic_fronts()
    distance_bins = [0, max_attach_distance, max_allowed_sideways_reach]
    
    program = ProgramTree()
    for reference_object_idx, object in enumerate(room.objects):
        subprogram = ProgramTree()

        distance, side = query_object.distance(object)
        distance_binned = np.digitize(distance, distance_bins)

        # Add possible locations 
        if distance_binned == 2 and object.holds_humans: # Close enough to be reachable, but not close enough to be attached
            constraint = [
                constraint_types_map['reachable_by_arm'],
                query_object_idx,
                reference_object_idx,
                side
            ]
            subprogram.from_constraint(constraint) 
        elif distance_binned == 1: # close enough to be attached 
            constraint = [
                constraint_types_map['attach'],
                query_object_idx,
                reference_object_idx,
                side
            ]
            subprogram.from_constraint(constraint) 
        
        # Constrict possible orientations 
        object_semantic_fronts = object.world_semantic_fronts()
        overlap = query_semantic_fronts.intersection(object_semantic_fronts)
        if len(overlap) and len(subprogram): # Algin, object points in the same direction 
            constraint = [
                constraint_types_map['align'],
                query_object_idx,
                reference_object_idx,
                direction_types_map['<pad>']
            ]
            
            other_tree = ProgramTree()
            other_tree.from_constraint(constraint)
            subprogram.combine('and', other_tree)
        else: # face, not possible for query object to both face and be aligned with object 
            if query_object.front_facing and len(subprogram):
                front_facing_direction = list(query_semantic_fronts)[0]
                if abs(front_facing_direction - side) == 2:
                    constraint = [
                        constraint_types_map['face'],
                        query_object_idx,
                        reference_object_idx,
                        direction_types_map['<pad>']      
                    ]
                    other_tree = ProgramTree()
                    other_tree.from_constraint(constraint)
                    subprogram.combine('and', other_tree)
        
        if len(subprogram):
            # Constrict possible locations and orientations to match this subprogram 
            if len(program): # take this subprogram, and it with the final output 
                program.combine('and', subprogram)
            else:
                program = subprogram

    # If program is empty (align, wall) -> place anywhere in any orientation 
    if not len(program):
        constraint = [
            constraint_types_map['align'],
            query_object_idx,
            0,
            direction_types_map['<pad>']
        ]
        program.from_constraint(constraint)

    return program

def extract_programs(scene_list):
    xs = []
    ys = []
    for scene in scene_list:
        for subscene, query_object in scene.permute():
            program = generate_most_restrictive_program(subscene, query_object)
            # Convert scene to object list and program to structures + constraints
            program_tokens = program.to_tokens()
            query_object_vector = query_object.vectorize(subscene.objects[0])
            query_object_vector[4] = 0
            query_object_vector[5] = 0
            subscene_vector = np.append(
                subscene.vectorize(),
                query_object_vector,
                axis = 0
            )
            xs.append(subscene_vector)
            ys.append(program_tokens)
    
    write_program_data(xs, ys)

def write_program_data(xs, ys):
    for objects_vector, program_tokens in zip(xs, ys):
        structure_sequence = program_tokens['structure']
        constraint_sequence = program_tokens['constraints']

def read_program_data():
    pass