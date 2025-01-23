


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