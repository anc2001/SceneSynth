from main.common import utils
from main.common.object_base import SceneObject, BBox, LineSeg
from main.config import colors, direction_types_map, num_angles

import numpy as np

def get_object(*args, **kwargs):
    """
    """
    type = args[0]
    if type == 'wall':
        scene = args[1]
        walls = args[2]

        info = {}
        info['color'] = colors['wall']
        info['id'] = 0
        info['holds_humans'] = 0
        info['semantic_fronts'] = [0, 1, 2, 3]

        return Wall(info, scene, walls)
    elif type == 'furniture':
        object_info = args[1]

        model_info = object_info['model_info']
        info = {}

        valid = False
        if model_info['super_category'] == 'bed':
            info['color'] = colors['bed']
            info['id'] = 1
            info['holds_humans'] = True
            info['semantic_fronts'] = {1}
            valid = True
        elif model_info['super_category'] == 'chair':
            info['color'] = colors['chair']
            info['id'] = 5
            info['holds_humans'] = True
            info['semantic_fronts'] = {1}
            valid = True
        elif model_info['super_category'] == 'cabinet/shelf/desk':
            if model_info['category'] == 'wardrobe':
                info['color'] = colors['wardrobe']
                info['id'] = 2
                info['holds_humans'] = False
                info['semantic_fronts'] = {1}
                valid = True
            elif model_info['category'] == 'nightstand':
                info['color'] = colors['nightstand']
                info['id'] = 3
                info['holds_humans'] = False
                info['semantic_fronts'] = {1}
                valid = True
        elif model_info['super_category'] == 'table':
            info['color'] = colors['desk']
            info['id'] = 4
            info['holds_humans'] = False
            info['semantic_fronts'] = {0, 1, 2, 3}
            valid = True
        
        if valid:
            info['object_info'] = object_info
            return Furniture(info)
        else:
            return None
    else:
        print("Invalid type")
        return None  

class Furniture(SceneObject):
    def __init__(self, info) -> None:
        super().__init__(info)
        self.bbox = BBox(info['object_info']['size'])
        self.center = self.bbox.center
        self.extent = self.bbox.extent
        
        # Line Segs 
        line_seg_indices = [
            [1, 3], # Right 
            [0, 1], # Up
            [2, 0], # Left
            [2, 3] # Down 
        ]

        line_seg_normals = [
            [1, 0, 0], # Right 
            [0, 0, 1], # Up
            [-1, 0, 0], # Left
            [0, 0, -1] # Down 
        ]

        line_segs = []
        for i in range(len(line_seg_indices)):
            indices = line_seg_indices[i]
            normal = line_seg_normals[i]
            point1 = self.bbox.vertices[indices[0]]
            point2 = self.bbox.vertices[indices[1]]
            line_seg = LineSeg(point1, point2, normal)
            line_segs.append(line_seg)
        self.line_segs = np.array(line_segs)

        self.rotate(- info['object_info']['rotation'])
        self.translate(info['object_info']['translation'])

    def rotate(self, theta):
        """
        theta : float of rotation given in radians 
        """
        self.bbox.rotate(theta)
        for line_seg in self.line_segs:
            line_seg.rotate(theta)

    def translate(self, amount : np.ndarray):
        """
        amount : np.ndarray of shape (3,)
        """
        self.bbox.translate(amount)
        for line_seg in self.line_segs:
            line_seg.translate(amount) 
    
    def vectorize(self, wall_object):
        """
        (category, holds_humans, size, position, rotation) - size and position in 2D
        """
        center = self.center - wall_object.center # center around floor 
        return np.array([[
            self.id, 
            self.holds_humans, 
            self.extent[0], 
            self.extent[2], 
            center[0], 
            center[2], 
            self.bbox.rot
        ]])

    def write_to_image(self, scene, image):
        # Write bounding box to image 
        for face in self.bbox.faces:
            triangle = self.bbox.vertices[face]
            utils.write_triangle_to_image(triangle, scene, image, self.color)

        # Write triangle directions to image 
        base_length = scene.cell_size * 8
        height = (np.sqrt(3) * base_length) / 2 # Equilateral triangle 
        for direction, segment in enumerate(self.line_segs):
            triangle_color = colors['directions'][direction]
            segment_centroid = segment.centroid()
            segment_normal = np.array(segment.normal)
            segment_vector = utils.normalize(segment.p2 - segment.p1)

            a = segment_centroid + (segment_vector * base_length / 2)
            b = segment_centroid - (segment_vector * base_length / 2)
            c = segment_centroid + segment_normal * height
            triangle = [a, b, c]
            utils.write_triangle_to_image(triangle, scene, image, triangle_color)

    def distance(self, reference : SceneObject):
        """
        Calculates the minimum distance between the this and the given object 
        Distance value of 0 means that the two objects intersect or overlap 

        returns distance, direction
        """        
        min_distance = np.finfo(np.float64).max 
        direction_vector = np.zeros(3)
        for object_line_seg in self.line_segs:
            for reference_line_seg in reference.line_segs:
                min_distance_tuple = object_line_seg.distance(reference_line_seg)
                if min_distance_tuple[0] < min_distance:
                    min_distance = min_distance_tuple[0]
                    if min_distance == 0:
                        direction_vector = min_distance_tuple[1] - reference.center
                    else:
                        direction_vector = min_distance_tuple[1] - min_distance_tuple[2]
                elif min_distance == min_distance_tuple[0]:
                    if min_distance == 0:
                        direction_vector += min_distance_tuple[1] - reference.center
                    else:
                        direction_vector += min_distance_tuple[1] - min_distance_tuple[2]       
        
        direction_vector = utils.normalize(direction_vector)
        side = reference.point_to_side(reference.center + direction_vector)
        return min_distance, side

    def world_semantic_fronts(self):
        """
        returns the semantic fronts of the object in world space
        """
        line_segs = self.line_segs[list(self.semantic_fronts)]
        angle_indices = []
        for line_seg in line_segs:
            angle_idx = utils.vector_angle_index(np.array([1,0,0]), line_seg.normal)
            angle_indices.append(angle_idx)
        return set(angle_indices)
     
    def line_segs_in_direction(self, direction, world_space = True):
        """
        returns line_segs with normal that points in direction 'direction'
        world_space indicates whether the direction is given world space
            or otherwise relative to the local coordinate from of the object 
        """
        if world_space:
            line_segs = []
            for line_seg in self.line_segs:
                angle_idx = utils.vector_angle_index(np.array([1,0,0]), line_seg.normal)
                if angle_idx == direction:
                    line_segs.append(line_seg)
            return line_segs
        else:
            return self.line_segs[direction]
    
    def point_inside(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)
        """
        return self.bbox.point_inside(point)
    
    def point_to_side(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)

        returns local index of corresponding side 
        """
        return self.bbox.point_to_side(point)
    
    def local_direction_to_world(self, angle):
        return angle + utils.angle_to_index(self.bbox.rot) % num_angles

class Wall(SceneObject):
    def __init__(self, info, scene, walls) -> None:
        super().__init__(info)
        self.line_segs = []
        for wall in walls:
            for triangle in scene.faces:
                if wall[0] in triangle and wall[1] in triangle:
                    # Do a vector projection from the other vertex onto the wall side 
                    otherPoint = None
                    for point in triangle:
                        if not point == wall[0] and not point == wall[1]:
                            otherPoint = point
                    
                    two_points = scene.vertices[wall]
                    line_seg = LineSeg(two_points[0], two_points[1], None)
                    point_to = scene.vertices[otherPoint]
                    normal = line_seg.normal_to_point(point_to)
                    line_seg.normal = normal
                    self.line_segs.append(line_seg)
        
        self.vertices = scene.vertices
        self.faces = scene.faces
        self.center = np.mean(self.vertices, axis = 0)
        self.extent = np.amax(self.vertices, axis = 0) - np.amin(self.vertices, axis = 0)

    def vectorize(self):
        """
        (category, holds_humans, size, position, rotation) - size and position in 2D
        """
        return np.array([[self.id, False, self.extent[0], self.extent[2], 0, 0, 0]])

    def write_to_image(self, scene, image):
        pass
    
    def world_semantic_fronts(self):
        """
        returns the semantic fronts of the object in world space
        """
        return set(self.semantic_fronts)
     
    def line_segs_in_direction(self, direction, world_space = True):
        """
        returns line_segs with normal that points in direction 'direction'
        world_space indicates whether the direction is given world space
            or otherwise relative to the local coordinate from of the object 
        """
        # There is no difference between world_space and local space for wall, because the local coordinate frame is the world space! 
        line_segs = []
        for line_seg in self.line_segs:
            angle_idx = utils.vector_angle_index(np.array([1,0,0]), line_seg.normal)
            if angle_idx == direction:
                line_segs.append(line_seg)
        return line_segs
    
    def point_inside(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)
        """
        for face in self.faces:
            if utils.point_triangle_test(point, self.vertices[face]):
                return True
        return False
    
    def point_to_side(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)

        returns local index of corresponding side 
        """
        return direction_types_map['<pad>']
    
    def local_direction_to_world(self, angle):
        return angle