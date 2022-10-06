from main.common.utils import write_triangle_to_image, powerset
from main.common.object import Furniture, get_object
from main.config import data_filepath, grid_size, colors

import numpy as np
import open3d as o3d
import pickle
import os 

def get_scene_list():
    scene_list = np.array([])
    with open(os.path.join(data_filepath, 'kai_parse.pkl'), 'rb') as f:
        room_info_list = pickle.load(f)
        for room_info in room_info_list:
            scene = Scene(room_info = room_info)
            scene_list = np.append(scene_list, scene)
    return scene_list

class Scene():
    """
    List of active member variables

    self.objects : np.ndarray of the objects in the room, first is always the walls 
    self.vertices : np.ndarray - (N, 3) array of vertices of the floor mesh
    self.faces : np.ndarray - (M, 3) array of faces of the floor mesh
    self.cell_size : float - the discretization of the grid 
    self.corner_pos : np.ndarray - (3,) the top left corner of the grid 
    """
    def __init__(self, room_info : dict = None) -> None:
        if room_info:
            self.init_room_geometry(
                room_info['floor_plan']['vertices'], 
                room_info['floor_plan']['faces']
            )

            for object_info in room_info['objects']:
                object = get_object('furniture', object_info, False)
                if object:
                    self.objects = np.append(self.objects, object)
    
    def init_room_geometry(self, vertices : np.ndarray, faces : np.ndarray) -> None:
        """
        Initializes the floor mesh of the scene
        Sets
            self.vertices, self.faces, self.walls, self.wall_directions 
        """
        room_mesh = o3d.geometry.TriangleMesh()
        room_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
        room_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
        room_mesh.merge_close_vertices(0.02)
        room_mesh.remove_duplicated_vertices()
        room_mesh.remove_duplicated_triangles()
        room_mesh.remove_degenerate_triangles()
        self.vertices = np.asarray(room_mesh.vertices)
        self.faces = np.asarray(room_mesh.triangles)

        walls = np.asarray(room_mesh.get_non_manifold_edges(allow_boundary_edges= False))
        wall_object = get_object('wall', self, walls)
        self.objects = np.array([wall_object])

        # Calculate the min corner position and cell size self floor mesh
        min_bound = np.amin(self.vertices, axis = 0)
        max_bound = np.amax(self.vertices, axis = 0)
        center = np.mean([min_bound, max_bound], axis = 0)

        total_floor_extent = max_bound - min_bound
        largest_dim_idx = np.argmax(total_floor_extent)
        largest_dim = total_floor_extent[largest_dim_idx] + 0.2

        # Calculate cell size and corner position so that all sides of the self are padded 
        self.cell_size = largest_dim / grid_size
        self.corner_pos = center - [largest_dim / 2, 0, largest_dim / 2]   

    def copy(self, empty=False):
        new_scene = Scene()
        new_scene.vertices = np.array(self.vertices)
        new_scene.faces = np.array(self.faces)
        new_scene.cell_size = self.cell_size
        new_scene.corner_pos = self.corner_pos
        if empty:
            new_scene.objects = np.array(self.objects[:1])
        else:
            new_scene.objects = np.array(self.objects)
        return new_scene
    
    def add_object(self, obj : Furniture, inplace=True):
        # Maintain canonical ordering
        if inplace:
            self.objects = np.append(self.objects, obj)
        else:
            new_scene = self.copy()
            new_scene.add_object(obj, inplace=True)
            return new_scene

    def remove_object(self, index : int, inplace=True):
        if inplace:
            new_object_list = self.objects[:index] + self.objects[index+1:]
            if self.grid_active:
                pass
        else:
            new_scene = self.copy()
            new_scene.remove_object(index, inplace=True)
            return new_scene

    def permute(self):
        # Given the current objects, return a list of (scene, object) tuples
        scene_object_pairs = []
        empty_scene = self.copy(empty=True) # includes wall
        rest_objects = self.objects[1:]
        rest_objects_indices = set(range(len(rest_objects)))
        possibilities = list(powerset(rest_objects_indices))
        for obj_idx_tuple in possibilities[:-1]: # disinclude the last set of all indices 
            objects_in_room = rest_objects[list(obj_idx_tuple)]
            possible_query_object_indices = rest_objects_indices.difference(set(obj_idx_tuple))
            for query_object_idx in possible_query_object_indices:
                query_object = rest_objects[query_object_idx]
                new_scene = empty_scene.copy()
                new_scene.objects = np.append(new_scene.objects, objects_in_room)
                scene_object_pairs.append((new_scene, query_object))
        
        return scene_object_pairs

    def vectorize(self):
        object_list = np.array([])
        for object in self.objects:
            if len(object_list):
                object_list = np.append(
                    object.vectorize(self.objects[0]), 
                    object_list, 
                    axis = 0
                )
            else:
                object_list = object.vectorize()
                
        return object_list
    
    def print(self, image):
        """
        Prints the orthographic view 
        """
        image[:, :, :] = colors['outside']

        # Mask all points inside 
        for face in self.faces:
            triangle = self.vertices[face]
            write_triangle_to_image(triangle, self, image, colors['inside'])
        
        for object in self.objects:
            object.write_to_image(self, image)
        
        image = np.rot90(image, axes=(0,1))