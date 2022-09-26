# Ability to return return variations of current self
#   Given the objects, return a list of (self, object) tuples

import numpy as np
import open3d as o3d
import matplotlib.image as img

from main.common import config
from main.common.object import Furniture

class Scene:
    """
    List of active member variables

    self.vertices : np.ndarray - (N, 3) array of vertices of the floor mesh
    self.faces : np.ndarray - (M, 3) array of faces of the floor mesh
    self.walls : np.ndarray - (P, 2) array of walls of the floor mesh
    self.wall_directions : np.ndarray - (P, ) array of wall directions of the floor mesh

    self.grid_active : bool - Whether the grid is active or not
    self.grid : np.ndarray - (GRID_SIZE, GRID_SIZE, ) array of the grid
        At every point is essentially a struct with the following fields:
            0: category - int [-1 outside, 0 empty, 1 object]
            1: canonical direction - int
            
    """
    def __init__(self) -> None:
        pass
    
    def init_floor_mesh(self, vertices : np.ndarray, faces : np.ndarray) -> None:
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
        new_vertices = np.asarray(room_mesh.vertices)
        new_faces = np.asarray(room_mesh.triangles)
        walls = np.asarray(room_mesh.get_non_manifold_edges(allow_boundary_edges= False))

        wall_normals_binned = []
        for wall in walls:
            for triangle in new_faces:
                if wall[0] in triangle and wall[1] in triangle:
                    # Do a vector projection from the other vertex onto the wall side 
                    otherPoint = None
                    for point in triangle:
                        if not point == wall[0] and not point == wall[1]:
                            otherPoint = point
                    
                    line_seg = new_vertices[wall]
                    point_to = new_vertices[otherPoint]
                    normal = utils.line_seg_normal(point_to, line_seg)
                    angle_idx = utils.vector_angle_index([1,0,0], normal)
                    wall_normals_binned.append(angle_idx)
        
        self.vertices = new_vertices
        self.faces = new_faces
        self.walls = walls
        self.wall_directions = np.array(wall_normals_binned)  

    def copy(self, empty=False):
        new_scene = Scene()
        new_scene.vertices = np.array(self.vertices)
        new_scene.faces = np.array(self.faces)
        if self.grid_active:
            new_scene.grid_active = True
            new_scene.orthographic_view = np.array(self.orthographic_view)
        if not empty:
            new_scene.objects = np.array(self.objects)
        return new_scene
    
    def add_object(self, obj : Furniture, inplace=True):
        # Maintain canonical ordering
        if inplace:
            pass
            if self.grid_active:
                pass
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

    def backtrace(self) -> list:
        # Given the current objects, return a list of (Scene, object) tuples
        pass
    
    def init_grid(self):
        """
        This single initialization method will 
            - set the min corner position and cell size 
            - encode walls and present objects
            - mask out grid cells outside of the floor mesh with -1
        """
        self.grid_active = True

        GRID_SIZE = config['Language']['grid_size']
        self.orthographic_view = - np.ones((GRID_SIZE, GRID_SIZE))

        # Calculate the min corner position and cell size self floor mesh
        min_bound = np.amin(self.vertices, axis = 0)
        max_bound = np.amax(self.vertices, axis = 0)

        center = np.mean([min_bound, max_bound], axis = 0)
        total_floor_extent = max_bound - min_bound
        largest_dim_idx = np.argmax(total_floor_extent)
        largest_dim = total_floor_extent[largest_dim_idx] + 0.2

        # Calculate cell size and corner position so that all sides of the self are padded 
        self.cell_size = largest_dim / GRID_SIZE
        self.corner_pos = center - [largest_dim / 2, 0, largest_dim / 2]
        
        # Mask out grid cells inside each triangle 
        for face in self.faces:
            current_triangle = self.vertices[face]
            face_min_bound = np.amin(current_triangle, axis = 0)
            face_max_bound = np.amax(current_triangle, axis = 0)
            grid_min_bound, grid_max_bound = utils.get_grid_from_bounds(face_min_bound, face_max_bound, self)
            for i in range(grid_min_bound[0], grid_max_bound[0] + 1):
                for j in range(grid_min_bound[2], grid_max_bound[2] + 1):
                    cell_center = self.corner_pos + np.array([i, 0, j]) * self.cell_size
                    cell_center_2d = [cell_center[0], cell_center[2]]
                    vertices_2d = [[vertex[0], vertex[2]] for vertex in current_triangle]
                    if utils.point_triangle_test(cell_center_2d, vertices_2d):
                        self.orthographic_view[i, j ] = 0
        
        only_walls = np.zeros(self.orthographic_view.shape)
        for i in range(GRID_SIZE):
            horz_wall_indices = utils.first_and_last_seq(self.orthographic_view[i, :], -1)
            for switch, idx in enumerate(horz_wall_indices):
                if not idx == 0 and not idx == GRID_SIZE - 1:
                    only_walls[i, idx] = 1
                    angle_idx = 1 if switch % 2 else 3
            
            vert_wall_indices = utils.first_and_last_seq(self.orthographic_view[:, i], -1)
            for switch, idx in enumerate(vert_wall_indices):
                if not idx == 0 and not idx == GRID_SIZE - 1:
                    only_walls[idx, i] = 1
                    angle_idx = 0 if switch % 2 else 2
        self.orthographic_view[only_walls == 1] = 1

        self.empty_orthographic_view = np.array(self.orthographic_view)

        for i, object in enumerate(self.objects):
            value_to_write = 2 + i
            grid_min_bound, grid_max_bound = object.aabb_grid_bounds(self)
            for i in range(grid_min_bound[0], grid_max_bound[0]):
                for j in range(grid_min_bound[2], grid_max_bound[2]):
                    cell_center = self.corner_pos + np.array([i + 0.5, 0, j + 0.5]) * self.cell_size
                    if object.point_inside(cell_center):
                        self.orthographic_view[i, j] = value_to_write
    
    def print_room(self, filepath):
        """
        Prints the orthographic view and self encoding of the self
        """
        GRID_SIZE = config['Language']['grid_size']
        COLORS = config['Colors']
        if not self.grid_active:
            self.init_grid()
        
        image = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.orthographic_view[i, j] == -1:
                    image[i, j, :] = COLORS['outside'] 
                elif self.orthographic_view[i, j] == 0:
                    image[i, j, :] = COLORS['inside'] 
                elif self.orthographic_view[i, j] == 1:
                    image[i, j, :] = COLORS['wall'] 
                else:
                    object = self.objects[int(self.orthographic_view[i, j] - 2)]
                    image[i, j, :] = object.color
        
        base_length = self.cell_size * 8
        height = (np.sqrt(3) * base_length) / 2
        for object in self.objects:
            for direction, segment_indices in enumerate(object.line_segs):
                segment = object.vertices[segment_indices]
                triangle_color = COLORS['directions'][direction]
                segment_centroid = np.mean(segment, axis = 0)
                segment_normal = object.line_seg_normals[direction]
                segment_vector = segment[1] - segment[0]
                segment_vector = segment_vector / (np.linalg.norm(segment_vector) + 1e-8)

                a = segment_centroid + (segment_vector * base_length / 2)
                b = segment_centroid - (segment_vector * base_length / 2)
                c = segment_centroid + segment_normal * height
                triangle = [a, b, c]
                triangle_2d = [[vertex[0], vertex[2]] for vertex in triangle]
                min_bound = np.amin(triangle, axis = 0)
                max_bound = np.amax(triangle, axis = 0)
                grid_min_bound, grid_max_bound = utils.get_grid_from_bounds(min_bound, max_bound, self)
                for i in range(grid_min_bound[0], grid_max_bound[0]):
                    for j in range(grid_min_bound[2], grid_max_bound[2]):
                        center_point = self.corner_pos + np.array([i + 0.5, 0, j + 0.5]) * self.cell_size
                        center_point_2d = np.array([center_point[0], center_point[2]])
                        if utils.point_triangle_test(center_point_2d, triangle_2d):
                            image[i, j, :] = triangle_color
        
        image = np.rot90(image, axes=(0,1))
        img.imsave(filepath, image)