from main.common import config
from main.common import utils
from main.common.object_base import SceneObject, BBox, LineSeg

import numpy as np

def get_object(*args, **kwargs):
    """
    """
    type = args[0]
    if type == 'wall':
        scene = args[1]
        walls = args[2]

        info = {'query_object' : False}
        info['color'] = config['Colors']['wall']
        info['id'] = 0
        info['holds_humans'] = False
        info['semantic_fronts'] = [0, 1, 2, 3]

        return Wall(info, scene, walls)
    elif type == 'furniture':
        object_info = args[1]
        is_query_object = args[2]

        model_info = object_info['model_info']
        info = {'query_object' : is_query_object}
        if model_info['super_category'] == 'bed':
            info['color'] = config['Colors']['bed']
            info['id'] = 1
            info['holds_humans'] = True
            info['semantic_fronts'] = [1]
        elif model_info['super_category'] == 'chair':
            info['color'] = config['Colors']['chair']
            info['id'] = 5
            info['holds_humans'] = True
            info['semantic_fronts'] = [1]
        elif model_info['super_category'] == 'cabinet/shelf/desk':
            if model_info['category'] == 'wardrobe':
                info['color'] = config['Colors']['wardrobe']
                info['id'] = 2
                info['holds_humans'] = False
                info['semantic_fronts'] = [1]
            elif model_info['category'] == 'nightstand':
                info['color'] = config['Colors']['nightstand']
                info['id'] = 3
                info['holds_humans'] = False
                info['semantic_fronts'] = [1]
        elif model_info['super_category'] == 'table':
            info['color'] = config['Colors']['desk']
            info['id'] = 4
            info['holds_humans'] = False
            info['semantic_fronts'] = [0, 1, 2, 3]
        
        info['object_info'] = object_info
        return Furniture(info)
    else:
        print("Invalid type")
        return None  

class Furniture(SceneObject):
    """
    self.query_object states whether the object is a query object or not 
    """
    def __init__(self, info) -> None:
        super().__init__(info)
        self.bbox = BBox(info['object_info']['size'])
        
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

        self.line_segs = []
        for i in range(len(line_seg_indices)):
            indices = line_seg_indices[i]
            normal = line_seg_normals[i]
            point1 = self.vertices[indices[0]]
            point2 = self.vertices[indices[1]]
            line_seg = LineSeg(point1, point2, normal)
            self.line_segs.append(line_seg)

        self.rotate(- info['object_info']['rotation'])
        self.translate(info['object_info']['translation'])

    def rotate(self, theta : float):
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

    def write_to_image(self, scene, image):
        """
        Masks the current object in the room  
        """
        # Write bounding box to image 
        for face in self.bbox.faces:
            triangle = self.bbox.vertices[face]
            utils.write_triangle_to_image(triangle, scene, image, self.color)

        # Write triangle directions to image 
        base_length = self.cell_size * 8
        height = (np.sqrt(3) * base_length) / 2 # Equilateral triangle 
        for direction, segment_indices in enumerate(object.line_segs):
            segment = object.vertices[segment_indices]
            triangle_color = config['colors']['directions'][direction]
            segment_centroid = np.mean(segment, axis = 0)
            segment_normal = object.line_seg_normals[direction]
            segment_vector = segment[1] - segment[0]
            segment_vector = segment_vector / (np.linalg.norm(segment_vector) + 1e-8)

            a = segment_centroid + (segment_vector * base_length / 2)
            b = segment_centroid - (segment_vector * base_length / 2)
            c = segment_centroid + segment_normal * height
            triangle = [a, b, c]
            utils.write_triangle_to_image(triangle, scene, image, triangle_color)

    # Only called when the object is a query object 
    def distance(self, reference):
        """
        Calculates the minimum distance between the this and the given object 
        Distance value of 0 means that the two objects intersect or overlap 

        returns distance, direction
        """
        if not self.query_object:
            print("Called distance on non query object")
            exit()
        
        min_distance = [(np.finfo(np.float64).max, None, None)] # List to include points that tie 
        for object_line_seg in self.line_segs:
            for reference_line_seg in reference.line_segs:
                min_distance_tuple = object_line_seg.distance(reference_line_seg)
                if min_distance[0][0] == min_distance_tuple[0]:
                    min_distance.append(min_distance_tuple)
                elif min_distance[0][0] > min_distance_tuple[0]:
                    min_distance = [min_distance_tuple]
        
        # In case of two intersections 
        direction_vector = np.zeros(3)
        distance = min_distance[0][0]
        for point_combo in min_distance:
            vector = point_combo[1] - point_combo[2] # Point on object - Point on reference
            if distance == 0:
                vector = point_combo[1] - reference.center
            direction_vector += vector

        direction_vector = utils.normalize(direction_vector)
        local_direction = reference.bbox.point_to_side(reference.center + direction_vector)
        return distance, local_direction

    def world_semantic_fronts(self):
        """
        returns the semantic fronts of the object in world space
        """
        line_segs = self.line_segs[self.semantic_fronts]
        angle_indices = []
        for line_seg in line_segs:
            angle_idx = utils.vector_angle_index(line_seg.normal, np.array([1,0,0]))
            angle_indices.append(angle_idx)
        return angle_indices
     
    def line_segs_in_direction(self, direction, world_space = True):
        """
        returns line_segs with normal that points in direction 'direction'
        world_space indicates whether the direction is given world space
            or otherwise relative to the local coordinate from of the object 
        """
        if world_space:
            line_segs = []
            for line_seg in self.line_segs:
                angle_idx = utils.vector_angle_index(line_seg.normal, np.array([1,0,0]))
                if angle_idx == direction:
                    line_segs.append(line_seg)
            return line_segs
        else:
            return self.line_segs[direction]

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
                    
                    two_points = self.vertices[wall]
                    line_seg = LineSeg(two_points[0], two_points[1], None)
                    point_to = scene.vertices[otherPoint]
                    normal = line_seg.normal_to_point(point_to)
                    line_seg.normal = normal
                    self.line_segs.append(line_seg)

    def world_semantic_fronts(self):
        """
        returns the semantic fronts of the object in world space
        """
        return self.semantic_fronts
     
    def line_segs_in_direction(self, direction, world_space = True):
        """
        returns line_segs with normal that points in direction 'direction'
        world_space indicates whether the direction is given world space
            or otherwise relative to the local coordinate from of the object 
        """
        # There is no difference between world_space and local space for wall, because the local coordinate frame is the world space! 
        line_segs = []
        for line_seg in self.line_segs:
            angle_idx = utils.vector_angle_index(line_seg.normal, np.array([1,0,0]))
            if angle_idx == direction:
                line_segs.append(line_seg)
        return line_segs