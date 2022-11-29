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

def generate_most_restrictive_program(scene, query_object):
    query_object_idx = len(scene.objects)
    query_semantic_fronts = query_object.world_semantic_fronts()
    distance_bins = [0, max_attach_distance, max_allowed_sideways_reach]
    
    program = ProgramTree()

    # Special case wall 
    reference_object_idx = 0
    reference_object = scene.objects[0]
    sides = reference_object.infer_relation(query_object, distance_bins)
    for side in sides:
        constraint = [
            constraint_types_map['attach'],
            query_object_idx,
            reference_object_idx,
            side
        ]
        subprogram = ProgramTree()
        subprogram.from_constraint(constraint)
        program.combine('and', subprogram)
    overlap = query_semantic_fronts.intersection(sides)
    if len(overlap) and len(program):
        direction = list(query_semantic_fronts)[0]
        constraint = [
            constraint_types_map['align'],
            query_object_idx,
            reference_object_idx,
            direction
        ]
        subprogram = ProgramTree()
        subprogram.from_constraint(constraint)
        program.combine('and', subprogram)

    for reference_object_idx, reference_object in enumerate(scene.objects[1:]):
        reference_object_idx += 1
        subprogram = ProgramTree()

        distance = reference_object.distance(query_object)
        distance_binned = np.digitize(distance, distance_bins)
        if distance_binned == 1 or distance_binned == 2:
            side = reference_object.infer_relation(query_object)
        
        if distance_binned == 2 and reference_object.holds_humans and side >= 0:
            # Close enough to be reachable, but not close enough to be attached
            constraint_type = constraint_types_map['reachable_by_arm']
            constraint = [
                constraint_type,
                query_object_idx,
                reference_object_idx,
                side
            ]
            subprogram.from_constraint(constraint)
        elif distance_binned == 1 and side >= 0:
            # Close enough to be attached 
            constraint_type = constraint_types_map['attach']
            constraint = [
                constraint_type,
                query_object_idx,
                reference_object_idx,
                side
            ]
            subprogram.from_constraint(constraint)

        object_semantic_fronts = reference_object.world_semantic_fronts()
        overlap = query_semantic_fronts.intersection(object_semantic_fronts)
        if len(overlap) and len(subprogram): 
            # Align, object points in the same direction 
            if not reference_object.id == 0:
                direction = direction_types_map['<pad>']
            else:
                direction = list(query_semantic_fronts)[0]

            constraint = [
                constraint_types_map['align'],
                query_object_idx,
                reference_object_idx,
                direction
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
        
def extract_programs(scenes, query_objects):
    xs = [] # (scene, query_object) pairs
    ys = [] # programs 
    for scene, query_object in zip(scenes, query_objects):
        program = generate_most_restrictive_program(scene, query_object)
        program_tokens = program.to_tokens()
        if not verify_program(program_tokens, len(scene.objects)):
            print("Here!")
        xs.append((scene, query_object))
        ys.append(program_tokens)
    program_data = {
        "xs" : xs,
        "ys" : ys
    }
    return program_data