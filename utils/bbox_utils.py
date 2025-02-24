


def get_center_of_bbox(bbox):
    """
    Get the center of the bounding box
    
    Args:
        bbox (list): List of the bounding box coordinates
    
    Returns:
        list: List of the center of the bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
     
    """
    Calculate the width of a bounding box.
    Args:
        bbox (list or tuple): A list or tuple of four elements representing 
                              the bounding box in the format [x_min, y_min, x_max, y_max].
    Returns:
        float: The width of the bounding box.
    """
    # bbox[2] - bbox[0] is the same as x2 - x1 (because bbox = x1, y1, x2, y2)

    return bbox[2]-bbox[0]



def measure_distance(p1, p2):
    """
    Computes the Euclidean distance between two points in a 2D space.

    Args:
        p1 (tuple): The first point as (x1, y1).
        p2 (tuple): The second point as (x2, y2).

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    # Compute the squared difference in x-coordinates
    dx_squared = (p1[0] - p2[0]) ** 2

    # Compute the squared difference in y-coordinates
    dy_squared = (p1[1] - p2[1]) ** 2

    # Sum the squared differences and take the square root (Euclidean distance formula)
    return (dx_squared + dy_squared) ** 0.5
