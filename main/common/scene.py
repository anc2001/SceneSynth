from main.common.object import Furniture, get_object
from main.config import data_filepath, grid_size, colors
from main.common.mesh_to_mask import render_mesh

from itertools import chain, combinations
import numpy as np
import open3d as o3d

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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
            self.id = room_info['id']
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
        new_scene.id = self.id
        new_scene.vertices = np.array(self.vertices)
        new_scene.faces = np.array(self.faces)
        new_scene.cell_size = self.cell_size
        new_scene.corner_pos = self.corner_pos
        if empty:
            new_scene.objects = np.array(self.objects[:1])
        else:
            new_scene.objects = np.array(self.objects)
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
                    object_list, 
                    object.vectorize(self.objects[0]), 
                    axis = 0
                )
            else:
                object_list = object.vectorize()
                
        return object_list
    
    def convert_to_image(self, query_object=None, empty=False, with_query_object=False):
        """
        Prints the orthographic view 
        """
        image = np.zeros((grid_size, grid_size, 3))
        image[:, :, :] = colors['outside']

        if empty:
            wall = self.objects[0]
            wall.write_to_image(self.corner_pos, self.cell_size, image)
        else:
            for object in self.objects:
                object.write_to_image(self.corner_pos, self.cell_size, image)
            if with_query_object:
                query_object.write_to_image(self.corner_pos, self.cell_size, image)
        
        image = np.rot90(image, axes=(0,1))
        return image
    
    def convert_to_mask(self):
        mask = np.zeros((grid_size, grid_size))
        for object in self.objects:
            object.write_to_mask(self.corner_pos, self.cell_size, mask)
        return mask
    
    def check_if_objects_inside(self):
        wall_mask = np.zeros((grid_size, grid_size))
        self.objects[0].write_to_mask(self.corner_pos, self.cell_size, wall_mask)
        wall_mask = ~np.asarray(wall_mask, dtype = bool)
        for object in self.objects[1:]:
            object_mask = np.zeros((grid_size, grid_size))
            object.write_to_mask(self.corner_pos, self.cell_size, object_mask)
            disagreement = np.logical_and(wall_mask, object_mask)
            if np.sum(disagreement) > 1000:
                return False          
        return True

