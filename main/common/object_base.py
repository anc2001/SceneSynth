from main.common import utils

import numpy as np

class BBox():
    """
    BBox member variables
    self.extent - full size of object 
    self.vertices
    self.faces
    self.center
    self.bins
    self.bottom_right
    """
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
        self.center = np.array([0.0,0.0,0.0])

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
        self.rot = 0
    
    def rotate(self, theta):
        """
        theta : float of rotation given in radians 
        """
        self.rot += theta[0]
        if self.rot > 2 * np.pi:
            self.rot = 2 * np.pi - self.rot
        elif self.rot < 0:
            self.rot += 2 * np.pi
        
        rot_matrix = utils.get_rot_matrix(theta)
        self.vertices = np.matmul(self.vertices, rot_matrix)
        self.bottom_right = np.matmul(self.bottom_right, rot_matrix)

    def translate(self, amount : np.ndarray):
        """
        amount : np.ndarray of shape (3,)
        """
        amount[1] = 0
        self.vertices += amount
        self.center += amount

    def point_inside(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)
        """
        for face in self.faces:
            if utils.point_triangle_test(point,  self.vertices[face]):
                return True
        return False
    
    def point_to_side(self, point : np.ndarray):
        """
        point : np.ndarray of shape (3,)

        returns local index of corresponding side 
        """
        vector = utils.normalize(point - self.center)
        angle = np.arccos(np.dot(self.bottom_right, vector))
        plane_normal = np.cross(self.bottom_right, vector)
        if plane_normal[1] > 0:
            angle = 2 * np.pi - angle
        angle_idx = np.digitize(angle, self.bins) - 1
        return angle_idx 
        
class LineSeg():
    def __init__(self, point1, point2, normal) -> None:
        self.p1 = np.array(point1)
        self.p1[1] = 0
        self.p2 = np.array(point2)
        self.p2[1] = 0
        self.normal = np.array(normal)

    def rotate(self, theta):
        """
        theta : float of rotation given in radians 
        """
        rot_matrix = utils.get_rot_matrix(theta)
        self.p1 = np.matmul(self.p1, rot_matrix)
        self.p2 = np.matmul(self.p2, rot_matrix)
        self.normal = np.matmul(self.normal, rot_matrix)
        
    def translate(self, amount : np.ndarray):
        """
        amount : np.ndarray of shape (3,)
        """
        amount[1] = 0
        self.p1 += amount
        self.p2 += amount
    
    def calculate_centroid(self):
        return np.mean([self.p1, self.p2], axis = 0)
    
    def length(self):
        return np.linalg.norm(self.p1 - self.p2)
    
    def normal_to_point(self, point : np.ndarray):
        """
        Returns the perpindicular normal from the linseg that points towards point 
        c (point)
        |
        |   <- return this guy
        a ----- b
        """
        ab = self.p1 - self.p2
        ab_mag = np.linalg.norm(ab)
        ca = point - self.p1
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
        if delta == 0: return np.array([])
        u_a = ((p4[0] - p3[0]) * (p1[2] - p3[2]) - (p4[2] - p3[2]) * (p1[0] - p3[0])) / delta
        u_b = ((p2[0] - p1[0]) * (p1[2] - p3[2]) - (p2[2] - p1[2]) * (p1[0] - p3[0])) / delta
        intersection = (0 <= u_a <= 1) and (0 <= u_b <= 1)
        if intersection:
            point = p1 + u_a * (p2 - p1)
            return point
        else:
            return np.array([])

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
        point = self.intersect(line_seg)
        if len(point):
            return (0, point, point)

        distances = []
        # given_first arguments -> want the point on current line_seg first in order of arguments 
        distances.append(line_seg.distance_to_point(self.p1, given_first=True)) 
        distances.append(line_seg.distance_to_point(self.p2, given_first=True))
        distances.append(self.distance_to_point(line_seg.p1, given_first=False))
        distances.append(self.distance_to_point(line_seg.p2, given_first=False))
        distances = sorted(distances, key = lambda x : x[0])
        
        return distances[0]
    
    # def linearly_classify_points(self, points):
    #     """
    #     Treats lineseg as a linear classifier
    #     returns accumulated score of given points from classification
    #     """
    #     total_score = 0
    #     mid_point = self.calculate_centroid()
    #     for point in points:
    #         # check if point is within bounds

    #         ab = self.p1 - self.p2
    #         ab_mag = np.linalg.norm(ab)
    #         ca = point - self.p1
    #         projection = (np.dot(ca, ab) / (ab_mag ** 2)) * ab
    #         normal = ca - projection
    #         distance = np.linalg.norm(normal)
    #         multiplier = 1 if np.dot(self.normal, point - mid_point) > 0 else -1
    #         total_score += distance * multiplier
    #     return total_score
    
    def calculate_sub_area(self, points):
        """
        0 -- 1
        |    |
        2 -- 3
        """
        axis = 2 if self.p1[0] == self.p2[0] else 0
        min_bound = min(self.p1[axis], self.p2[axis])
        max_bound = max(self.p1[axis], self.p2[axis])
        
        sub_quad = np.array(points)
        for point in sub_quad:
            point[axis] = np.clip(point[axis], min_bound, max_bound)
        
        score = max(
            (sub_quad[0] - sub_quad[1])[axis],
            (sub_quad[0] - sub_quad[2])[axis]
        )

        return score
        
        