from main.common.language import ProgramTree
from main.config import \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types_map, constraint_types_map, \
    data_filepath

import numpy as np
from tqdm import tqdm
import pickle
import os

def generate_most_restrictive_program(room, query_object):
    query_object_idx = len(room.objects)
    query_semantic_fronts = query_object.world_semantic_fronts()
    distance_bins = [0, max_attach_distance, max_allowed_sideways_reach]
    
    program = ProgramTree()
    for reference_object_idx, reference_object in enumerate(room.objects):
        subprogram = ProgramTree()

        distance, side = query_object.distance(reference_object)
        distance_binned = np.digitize(distance, distance_bins)

        # Add possible locations 
        if distance_binned == 2 and reference_object.holds_humans: # Close enough to be reachable, but not close enough to be attached
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
        
        object_semantic_fronts = reference_object.world_semantic_fronts()
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
            if query_object.front_facing:
                front_facing_direction = list(query_semantic_fronts)[0]
                line_segs = query_object.line_segs_in_direction(front_facing_direction)
                for line_seg in line_segs:
                    ray_origin = line_seg.calculate_centroid()
                    ray = line_seg.normal
                    if reference_object.id: # not wall 
                        check, _ = reference_object.check_intersection(ray, ray_origin)
                        # Don't use second argument for now 
                        if check:
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
    for scene in tqdm(scene_list):
        for subscene, query_object in scene.permute():
            program = generate_most_restrictive_program(subscene, query_object)
            # Convert scene to object list and program to structures + constraints
            program_tokens = program.to_tokens()
            query_object_vector = query_object.vectorize(subscene.objects[0])
            query_object_vector[0, 4] = 0
            query_object_vector[0, 5] = 0
            subscene_vector = np.append(
                subscene.vectorize(),
                query_object_vector,
                axis = 0
            )
            xs.append(subscene_vector)
            ys.append(program_tokens)
    return xs, ys

def write_program_data(xs, ys):
    program_data = dict()
    program_data['xs'] = xs
    program_data['ys'] = ys
    filepath = os.path.join(data_filepath, 'program_data.pkl')
    with open(filepath, 'wb') as handle:
        pickle.dump(program_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_program_data():
    filepath = os.path.join(data_filepath, 'program_data.pkl')
    with open(filepath, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    
    return unserialized_data