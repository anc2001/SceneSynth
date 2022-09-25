# General utility functions for common objects 
import numpy as np

def normalize(vector):
    return vector / (np.linalg.norm(vector) + 1e-8)

def line_seg_normal(point, line_seg):
    """
    Returns the perpindicular normal from the linseg that points towards point 
    """
    a = line_seg[0]
    b = line_seg[1]
    c = point
    ab = a - b
    ab_mag = np.linalg.norm(ab)
    ca = c - a
    projection = (np.dot(ca, ab) / (ab_mag ** 2)) * ab
    normal = normalize(ca - projection)
    return normal

def closest_point_lineseg(p3, line_seg, given_first):
    """
    """
    p1 = line_seg[0]
    p2 = line_seg[1]
    line_seg_ray = p2 - p1
    u = (np.dot(p3, line_seg_ray) - np.dot(p1, line_seg_ray)) / np.dot(line_seg_ray, line_seg_ray)
    u = np.clip(u, 0, 1)
    p4 = p1 + u * line_seg_ray
    distance = np.linalg.norm(p3 - p4)
    minimum_tuple = (distance, p3, p4) if given_first else (distance, p4, p3)
    return minimum_tuple

def line_seg_intersect(line_seg1_2d, line_seg2_2d):
    """
    """
    p1 = line_seg1_2d[0]
    p2 = line_seg1_2d[1]
    p3 = line_seg2_2d[0]
    p4 = line_seg2_2d[1]
    delta = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if delta == 0: return (False, None)
    u_a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / delta
    u_b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / delta
    intersection = (0 <= u_a <= 1) and (0 <= u_b <= 1)
    if intersection:
        point = p1 + u_a * (p2 - p1)
        return (True, point)
    else:
        return (False, None)

def lineseg_distance(line_seg1_2d, line_seg2_2d):
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
    seg1 = np.array([[point[0], point[2]] for point in line_seg1_2d])
    seg2 = np.array([[point[0], point[2]] for point in line_seg2_2d])

    intersection_query = line_seg_intersect(seg1, seg2)
    if intersection_query[0]: 
        point1 = intersection_query[1]
        point1 = np.array([point1[0], 0, point1[1]])
        return (0, point1, point1)
    
    a = seg1[0]
    b = seg1[1]
    c = seg2[0]
    d = seg2[1]

    distances = []
    distances.append(closest_point_lineseg(a, seg2, given_first=True))
    distances.append(closest_point_lineseg(b, seg2, given_first=True))
    distances.append(closest_point_lineseg(c, seg1, given_first=False))
    distances.append(closest_point_lineseg(d, seg1, given_first=False))
    distances = sorted(distances, key = lambda x : x[0])
    
    point1 = distances[0][1]
    point1 = np.array([point1[0], 0, point1[1]])
    point2 = distances[0][2]
    point2 = np.array([point2[0], 0, point2[1]])
    return (distances[0][0], point1, point2)