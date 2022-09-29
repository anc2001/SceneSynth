from main.common.language import ProgramTree, constraint_types_map, direction_types_map
from main.config import max_allowed_sideways_reach

import numpy as np

def generate_most_restrictive_program(room, query_object):
    query_object_idx = len(room.objects)
    max_allowed_attach_distance = 0.1 * np.linalg.norm(query_object.bbox.size)
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
        object_semantic_fronts = object.semantic_fronts()
        overlap = query_semantic_fronts.intersect(object_semantic_fronts)
        if len(overlap) and len(subprogram): # Algin, object points in the same direction 
            constraint = [
                constraint_types_map['align'],
                query_object_idx,
                reference_object_idx,
                direction_types_map['<pad>']
            ]
            
            subprogram.combine('and', ProgramTree().from_constraint(constraint))
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
    for scene in scene_list:
        for subscene in scene.permute():
            pass
        # permute the scene 
        # 
        pass

def write_program_data(objects_list, programs_list):
    pass

def read_program_data():
    pass