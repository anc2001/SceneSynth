# Ability to return return variations of current self
#   Given the objects, return a list of (self, object) tuples

import numpy as np
import open3d as o3d
import matplotlib.image as img

from main.common import config, utils
from main.common.object import Furniture, get_object

class Scene():
    """
    List of active member variables

    self.objects : np.ndarray of the objects in the room, first is always the walls 
    self.vertices : np.ndarray - (N, 3) array of vertices of the floor mesh
    self.faces : np.ndarray - (M, 3) array of faces of the floor mesh
    self.cell_size : float - the discretization of the grid 
    self.corner_pos : np.ndarray - (3,) the top left corner of the grid 
    """
    def __init__(self, room_info : dict) -> None:
        self.init_room_geometry(
            room_info['floor_plan']['vertices'], 
            room_info['floor_plan']['faces']
        )
        for object_info in room_info['objects']:
            object = get_object('furniture', object_info, False)
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
        self.cell_size = largest_dim / config['Language']['grid_size']
        self.corner_pos = center - [largest_dim / 2, 0, largest_dim / 2]   

    def copy(self, empty=False):
        new_scene = Scene()
        new_scene.vertices = np.array(self.vertices)
        new_scene.faces = np.array(self.faces)
        new_scene.cell_size = self.cell_size
        new_scene.corner_pos = self.corner_pos
        if empty:
            new_scene.objects = self.objects[:1]
        else:
            new_scene.objects = self.objects
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

    def permute(self) -> list:
        # Given the current objects, return a list of (Scene, object) tuples
        pass
    
    def print(self, filepath):
        """
        Prints the orthographic view 
        """
        grid_size = config['Language']['grid_size']
        image = np.zeros((grid_size, grid_size, 3))
        image[:, :, :] = config['Colors']['outside']

        # Mask all points inside 
        for face in self.faces:
            triangle = self.vertices[face]
            utils.write_triangle_to_image(triangle, self, image, config['Colors']['inside'])
        
        for object in self.objects:
            object.write_to_image(self, image)
        
        image = np.rot90(image, axes=(0,1))
        img.imsave(filepath, image)