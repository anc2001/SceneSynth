class Furniture:
    def __init__(self) -> None:
        pass

class BBox:
    def __init__(self) -> None:
        pass

"""
What is required of the object 
- minimum distance from this object to another (and other info such as which side is the closest on both), 
- check if point is inside object 
- where the object is inside of the grid 
- in what world space direction does the object currently face 
- given another object whether it faces that object or not 
- the linesegs (and normals) that point in a particular world space direction 
- the lineseg (and normal) that corresponds with a given direction 

- basic transformations -> rotation, translation 
- size of object 
- vertices 
- faces 
"""