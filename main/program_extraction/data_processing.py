from main.common.language import ProgramTree
from main.config import max_allowed_sideways_reach, \
    direction_types_map, constraint_types_map

import numpy as np

def generate_most_restrictive_program(room, query_object):
    query_object_idx = len(room.objects)
    max_allowed_attach_distance = 0.1 * np.linalg.norm(query_object.bbox.extent)
    query_semantic_fronts = query_object.world_semantic_fronts()
    distance_bins = [0, max_allowed_attach_distance, max_allowed_sideways_reach]
    front_facing = len(query_semantic_fronts) == 1
    
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
            if front_facing:
                front_facing_direction = query_semantic_fronts[0]
                if abs(front_facing_direction - side) == 2:
                    constraint = [
                        constraint_types_map['face'],
                        query_object_idx,
                        reference_object_idx,
                        direction_types_map['<pad>']      
                    ]
                    subprogram.combine('and', ProgramTree().from_constraint(constraint))
        
        # Constrict possible locations and orientations to match this subprogram 
        if len(program): # take this subprogram, and it with the final output 
            program.combine('and', subprogram)
        else:
            program = subprogram

    # If program is empty (align, wall)
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
    objects = []
    programs = []
    for scene in scene_list:
        for subscene, query_object in scene.permute():
            program = generate_most_restrictive_program(subscene, query_object)
            # Convert scene to object list and program to structures + constraints
            program_tokens = program.to_tokens()
            subscene_vector = np.append(
                subscene.vectorize(), 
                query_object.vectorize(), 
                axis = 0
            )
            objects.append(subscene_vector)
            programs.append(program_tokens)
    
    write_program_data(objects, programs)

def write_program_data(objects_list, programs_list):
    pass

def read_program_data():
    pass