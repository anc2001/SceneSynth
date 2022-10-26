from main.common.language import ProgramTree
from main.config import \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types_map, constraint_types_map, \
    data_filepath, grid_size
from main.common.utils import angle_to_index
from main.common.language import verify_program

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

        if reference_object.id == 0:
            # wall
            distance, sides, accumulator = query_object.distance(reference_object, return_all = True)
            distance_binned = np.digitize(distance, distance_bins)
            other_possibilities = np.digitize(accumulator[:, 0], distance_bins)
            other_possibilities = accumulator[other_possibilities == distance_binned]
            for item in other_possibilities:
                sides.add(item[1])
            sides = sides.intersection(query_semantic_fronts)
        else:
            distance, sides = query_object.distance(reference_object)
            distance_binned = np.digitize(distance, distance_bins)

        # Add possible locations 
        for side in sides:
            if distance_binned == 2 and reference_object.holds_humans: 
                # Close enough to be reachable, but not close enough to be attached
                constraint = [
                    constraint_types_map['reachable_by_arm'],
                    query_object_idx,
                    reference_object_idx,
                    side
                ]
                other_tree = ProgramTree()
                other_tree.from_constraint(constraint)
                subprogram.combine('or', other_tree)
            elif distance_binned == 1: # close enough to be attached 
                constraint = [
                    constraint_types_map['attach'],
                    query_object_idx,
                    reference_object_idx,
                    side
                ]
                other_tree = ProgramTree()
                other_tree.from_constraint(constraint)
                subprogram.combine('or', other_tree)
        
        object_semantic_fronts = reference_object.world_semantic_fronts()
        overlap = query_semantic_fronts.intersection(object_semantic_fronts)
        if len(overlap) and len(subprogram): 
            # Algin, object points in the same direction 
            if not reference_object.id == 0:
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
            program.combine('and', subprogram)
    
    if not len(program):
        constraint = [
            constraint_types_map['align'],
            query_object_idx,
            0,
            direction_types_map['<pad>']
        ]
        program.from_constraint(constraint)
    return program

def verify_program_validity(program, scene, query_object):
    mask = program.evaluate(scene, query_object)
    valid_orientation = angle_to_index(query_object.bbox.rot)
    valid_placement = ((query_object.bbox.center - scene.corner_pos) / scene.cell_size).astype(int)
    x_range = np.clip(
        np.arange(valid_placement[0] - 3, valid_placement[0] + 3), 
        0, grid_size
    )
    y_range = np.clip(
        np.arange(valid_placement[2] - 3, valid_placement[2] + 3),
        0, grid_size
    )
    for i in x_range:
        for j in y_range:
            if mask[valid_orientation, i, j]:
                return True
    return False
        
def extract_programs(scene_list):
    xs = [] # (scene, query_object) pairs
    ys = [] # programs 
    for scene in tqdm(scene_list):
        for scene, query_object in scene.permute():
            program = generate_most_restrictive_program(scene, query_object)
            program_tokens = program.to_tokens()
            if not verify_program(program_tokens, len(scene.objects)):
                print("Here!")
            xs.append((scene, query_object))
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