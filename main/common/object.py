from main.common import utils
from main.common.object_base import BBox, LineSeg
from main.config import colors, direction_types_map, \
    num_angles, max_visibility_distance, \
    object_types_map, grid_size, colors

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
        info['id'] = object_types_map['wall']
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
            info['id'] = object_types_map['bed']
            info['holds_humans'] = True
            info['semantic_fronts'] = {1}
            valid = True
        elif model_info['super_category'] == 'chair':
            info['color'] = colors['chair']
            info['id'] = object_types_map['chair']
            info['holds_humans'] = True
            info['semantic_fronts'] = {1}
            valid = True
        elif model_info['super_category'] == 'cabinet/shelf/desk':
            if model_info['category'] == 'wardrobe':
                info['color'] = colors['wardrobe']
                info['id'] = object_types_map['wardrobe']
                info['holds_humans'] = False
                info['semantic_fronts'] = {1}
                valid = True
            elif model_info['category'] == 'nightstand':
                info['color'] = colors['nightstand']
                info['id'] = object_types_map['nightstand']
                info['holds_humans'] = False
                info['semantic_fronts'] = {1}
                valid = True
        elif model_info['super_category'] == 'table':
            info['color'] = colors['desk']
            info['id'] = object_types_map['desk']
            info['holds_humans'] = False
            info['semantic_fronts'] = {1}
            valid = True
        
        if valid:
            info['object_info'] = object_info
            return Furniture(info)
        else:
            return None
    else:
        print("Invalid type")
        return None  

class SceneObject():
    def __init__(self, info) -> None:
        self.id = info['id']
        self.color = info['color']
        self.holds_humans = info['holds_humans']
        self.semantic_fronts = info['semantic_fronts']
        self.front_facing = len(self.semantic_fronts) == 1

class Furniture(SceneObject):
    def __init__(self, info) -> None:
        super().__init__(info)
        self.bbox = BBox(info['object_info']['size'])
        self.center = self.bbox.center
        self.extent = self.bbox.extent
        self.init_line_segs()
        self.rotate(- info['object_info']['rotation'])
        self.translate(info['object_info']['translation'])
    
    def init_line_segs(self):
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

    def area(self):
        return self.bbox.area

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

    def write_to_image(self, scene_corner_pos, scene_cell_size, image, normalize = False):
        current_bbox = self.bbox
        temp_bbox = BBox(self.bbox.extent / 2)
        current_line_segs = self.line_segs
        if normalize:
            self.bbox = temp_bbox
            self.init_line_segs()
        
        # Write bounding box to image 
        verts = self.bbox.vertices
        faces = self.bbox.faces
        img = utils.render_orthographic(verts, faces, scene_corner_pos, scene_cell_size)
        image[np.asarray(img, dtype = bool)] = self.color

        # Write triangle directions to image 
        base_length = scene_cell_size * 8
        height = (np.sqrt(3) * base_length) / 2 # Equilateral triangle 
        for direction, segment in enumerate(self.line_segs):
            triangle_color = colors['directions'][direction]
            segment_centroid = segment.calculate_centroid()
            segment_normal = np.array(segment.normal)
            segment_vector = utils.normalize(segment.p2 - segment.p1)

            a = segment_centroid + (segment_vector * base_length / 2)
            b = segment_centroid - (segment_vector * base_length / 2)
            c = segment_centroid + segment_normal * height
            verts = np.array([a, b, c])
            faces = np.array([[0,1,2]])
            img = utils.render_orthographic(verts, faces, scene_corner_pos, scene_cell_size)
            image[np.asarray(img, dtype = bool)] = triangle_color
        
        if normalize:
            self.bbox = current_bbox
            self.line_segs = current_line_segs
    
    def write_to_mask(self, scene_corner_pos, scene_cell_size, mask):
        verts = self.bbox.vertices
        faces = self.bbox.faces
        img = utils.render_orthographic(verts, faces, scene_corner_pos, scene_cell_size)
        mask += img

    def distance(self, query : SceneObject):
        """
        Calculates the minimum distance between the this and the given object 
        Distance value of 0 means that the two objects intersect or overlap 

        returns distance, direction
        """
        min_distance = np.finfo(np.float64).max
        for query_line_seg in query.line_segs:
            for reference_line_seg in self.line_segs:
                min_distance_tuple = query_line_seg.distance(reference_line_seg)
                if min_distance_tuple[0] < min_distance:
                    min_distance = min_distance_tuple[0]

        return min_distance

    def infer_relation(self, query : SceneObject):
        side_to_return = -1
        max_coverage = 0
        for side, reference_line_seg in enumerate(self.line_segs):
            sub_length, area = reference_line_seg.calculate_sub_area(query.bbox.vertices)
            percent = sub_length / reference_line_seg.length()
            normal = reference_line_seg.normal
            score = 0
            for vertex in query.bbox.vertices:
                vec = utils.normalize(vertex - self.center)
                if np.dot(vec, normal) > 0:
                    score += 1
            if percent > 0.05 and sub_length > max_coverage and score == 4:
                side_to_return = side
                max_coverage = sub_length
        return side_to_return  
    
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
        return self.bbox.point_inside(point)
    
    def point_to_side(self, point : np.ndarray):
        return self.bbox.point_to_side(point)
    
    def local_direction_to_world(self, angle):
        return (angle + utils.angle_to_index(self.bbox.rot)) % num_angles

    def check_intersection(self, ray, ray_origin):
        emanating = LineSeg(
            ray_origin, 
            ray_origin + ray * 20,
            np.array([0,0,0])
        )
        to_sort = []
        for idx, line_seg in enumerate(self.line_segs):
            point = emanating.intersect(line_seg)
            if len(point):
                distance = np.linalg.norm(point - ray_origin)
                to_sort.append((distance, idx))
        
        if len(to_sort):
            min_distance_tuple = sorted(to_sort, key = lambda x : x[0])[0]
            return True, min_distance_tuple[1]
        else:
            return False, 4

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
        self.color = colors['inside']

    def vectorize(self):
        """
        (category, holds_humans, size, position, rotation) - size and position in 2D
        """
        return np.array([[self.id, False, self.extent[0], self.extent[2], 0, 0, 0]])
    
    def write_to_image(self, scene_corner_pos, scene_cell_size, image, normalize = False):
        img = utils.render_orthographic(self.vertices, self.faces, scene_corner_pos, scene_cell_size)
        image[np.array(img, dtype = bool)] = self.color

    def write_to_mask(self, scene_corner_pos, scene_cell_size, mask):
        img  = utils.render_orthographic(self.vertices, self.faces, scene_corner_pos, scene_cell_size)
        mask += img

    # Want to know all possible sides of the wall the object is attached to 
    def infer_relation(self, query, bins):
        sides = set()
        for query_line_seg in query.line_segs:
            for reference_line_seg in self.line_segs:
                sub_length, area = reference_line_seg.calculate_sub_area(query.bbox.vertices)
                percent = sub_length / reference_line_seg.length()
                min_distance_tuple = query_line_seg.distance(reference_line_seg)
                distance_binned = np.digitize(min_distance_tuple[0], bins)
                if distance_binned == 1 and percent > 0.05:
                    side = utils.vector_angle_index(
                        np.array([1,0,0]), 
                        reference_line_seg.normal
                    )
                    sides.add(side)
        
        return sides
    
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
        return direction_types_map['null']
    
    def local_direction_to_world(self, angle):
        return angle