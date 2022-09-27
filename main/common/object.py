from main.common import config
from main.common import utils

import numpy as np

def get_object(object_info):
    pass

"""
Unified Object API 

- minimum distance from this object to another (and other info such as which side is the closest on both), 

- Given grid, mask out where the object is inside it, infer whether color mode or not 

- in what world space direction does the object currently face 
- given another object whether it faces that object or not 

- the linesegs (and normals) that point in a particular world space direction 
- the lineseg (and normal) that corresponds with a given direction 
- check whether another object (bounding box) intersects with the current object 
- function which infers directional relation 


"""
class SceneObject():
    def __init__(self) -> None:
        self.color = None

    def write_to_array(self, grid):
        # Infer whether 2D or 3D grid 
        pass
    
    def point_inside(self, point):
        pass

    def bounds(self):
        """
        """
        pass

    def distance(self, object):
        """
        Calculates the minimum distance between the this and the given object 

        returns (distance, point_on_object, point_on_reference)
        """
        pass

class Furniture(SceneObject):
    def __init__(self, object_info) -> None:
        super().__init__()

class Wall(SceneObject):
    """
    - Need distance object is from closest wall -> also just a collection of linesegs, but normals pointing into the room 
    - This should be fine since object - object distance is just based on the closest lineseg   
    """
    def __init__(self, walls, wall_directions) -> None:
        super().__init__()

class BBox():
    def __init__(self, size) -> None:
        self.extent = 2 * size
        max_bound = size
        min_bound = -size
        """
        0 -- 1
        |    |
        2 -- 3
        """
        self.vertices = np.array([
            [min_bound[0], 0, max_bound[2]],
            [max_bound[0], 0, max_bound[2]],
            [min_bound[0], 0, min_bound[2]],
            [max_bound[0], 0, min_bound[2]]
        ])
        self.faces = np.array([[0, 3, 1], [0, 2, 3]])
        self.center = np.array([0,0,0])

        # Angular bins 
        v0 = utils.normalize(self.vertices[0])
        v1 = utils.normalize(self.vertices[1])
        v2 = utils.normalize(self.vertices[2])
        v3 = utils.normalize(self.vertices[3])
 
        angle_1 = np.arccos(np.dot(v3, v1))
        angle_2 = angle_1 + np.arccos(np.dot(v1, v0))
        angle_3 = angle_2 + np.arccos(np.dot(v0, v2))
        self.bins = [0, angle_1, angle_2, angle_3]
        self.bottom_right = utils.normalize(self.vertices[3])

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
    
    def rotate(self, theta : float):
        """
        theta : float of rotation given in radians 
        """
        rot_matrix = utils.get_rot_matrix(theta)
        self.vertices = np.matmul(self.vertices, rot_matrix)
        for line_seg in self.line_segs:
            line_seg.rotate(theta)

    def translate(self, amount : np.ndarray):
        """
        amount : np.ndarray of shape (3,)
        """
        self.vertices += amount
        self.center += amount
        for line_seg in self.line_segs:
            line_seg.translate(amount)

    def point_inside(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)
        """
        for face in self.faces:
            if utils.point_triangle_test(point,  self.vertices[face]):
                return True
        return False

class LineSeg():
    def __init__(self, point1, point2, normal) -> None:
        self.p1 = point1
        self.p2 = point2
        self.normal = normal

    def rotate(self, theta : float):
        """
        theta : float of rotation given in radians 
        """
        rot_matrix = utils.get_rot_matrix(theta)
        self.p1 = np.matmul(self.p1, rot_matrix)
        self.p2 = np.matmul(self.p2, rot_matrix)
        
    def translate(self, amount : np.ndarray):
        """
        amount : np.ndarray of shape (3,)
        """
        self.p1 += amount
        self.p2 += amount
    
    def normal_to_point(self, point : np.ndarray):
        """
        Returns the perpindicular normal from the linseg that points towards point 
        c 
        |
        |   <- return this guy
        a ----- b
        """
        ab = self.p1 - self.p2
        ab_mag = np.linalg.norm(ab)
        ca = point - self.point1
        projection = (np.dot(ca, ab) / (ab_mag ** 2)) * ab
        normal = utils.normalize(ca - projection)
        return normal
    
    def distance_to_point(self, p3 : np.ndarray, given_first : bool):
        """
        http://paulbourke.net/geometry/pointlineplane/
        p3  

        given_first - bool whether to return the given point first or the calculated point 

        p4 is a point on the line seg that is the closest to the given point p3 
        """
        line_seg_ray = self.p2 - self.p1
        u = (np.dot(p3, line_seg_ray) - np.dot(self.p1, line_seg_ray)) / np.dot(line_seg_ray, line_seg_ray)
        u = np.clip(u, 0, 1)
        p4 = self.p1 + u * line_seg_ray
        distance = np.linalg.norm(p3 - p4)
        minimum_tuple = (distance, p3, p4) if given_first else (distance, p4, p3)
        return minimum_tuple
    
    def intersect(self, line_seg):
        """
        http://paulbourke.net/geometry/pointlineplane/

        """
        p1 = self.p1
        p2 = self.p2
        p3 = line_seg.p1
        p4 = line_seg.p2
        delta = (p4[2] - p3[2]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[2] - p1[2])
        if delta == 0: return (False, None)
        u_a = ((p4[0] - p3[0]) * (p1[2] - p3[2]) - (p4[2] - p3[2]) * (p1[0] - p3[0])) / delta
        u_b = ((p2[0] - p1[0]) * (p1[2] - p3[2]) - (p2[2] - p1[2]) * (p1[0] - p3[0])) / delta
        intersection = (0 <= u_a <= 1) and (0 <= u_b <= 1)
        if intersection:
            point = p1 + u_a * (p2 - p1)
            return (True, point)
        else:
            return (False, None)

    def distance(self, line_seg):
        """
        Finds the minimum distance between two line segments in 2D and returns that distance as well as the respective points 
        With points AB and CD, calculate min distance as the minimum of the calculations 
            distance(A, CD)
            distance(B, CD)
            distance(C, AB)
            distance(D, AB)
        Returns (min_distance, point on line seg 1, point on line seg 2)

        Given points are 3D points with y = 0
        """
        intersection_query = self.intersect(self, line_seg)[0]
        if intersection_query[0]: 
            point1 = intersection_query[1]
            point1 = np.array([point1[0], 0, point1[1]])
            return (0, point1, point1)

        distances = []
        # given_first arguments -> want the point on current line_seg first in order of arguments 
        distances.append(line_seg.distance_to_point(self.p1, given_first=True)) 
        distances.append(line_seg.distance_to_point(self.p2, given_first=True))
        distances.append(self.distance_to_point(line_seg.p1, given_first=False))
        distances.append(self.distance_to_point(line_seg.p2, given_first=False))
        distances = sorted(distances, key = lambda x : x[0])
        
        return distances[0]