def calculate_iou_two_points(bbox1, bbox2):
    """
    Calculates intersection over union between two bounding boxes

    Assumes the bboxes have format [x1, y1, x2, y2]

    :param bbox1: A bounding box
    :param bbox2: A second bounding box
    :return: IoU
    """
    # Calculating corners of intersection box
    # Top left corner
    intersect_top_left_x = max(bbox1[0], bbox2[0])
    intersect_top_left_y = max(bbox1[1], bbox2[1])
    # Bottom right corner
    intersect_bottom_right_x = min(bbox1[2], bbox2[2])
    intersect_bottom_right_y = min(bbox1[3], bbox2[3])

    # We add +1 because the two boxes could be overlapping on one line of pixels (one edge), and
    # that shouldn't count as 0
    area_intersection = max(0, intersect_bottom_right_x - intersect_top_left_x + 1) * \
                        max(0, intersect_bottom_right_y - intersect_top_left_y + 1)

    area_bbox1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area_bbox2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = area_bbox1 + area_bbox2 - area_intersection

    return area_intersection / float(union_area)


def calculate_iou_one_point(bbox1, bbox2):
    """
    Calculates intersection over union between two bounding boxes. IoU = Area intersection / Area union.

    Assumes the bboxes have format [x, y, w, h]. That means that we only have the top left point and the width and
    height.

    Internally, this method just converts the bboxes to format [x1, y1, x2, y2] and delegates to execution to
    calculate_iou_two_points

    :param bbox1: A bounding box
    :param bbox2: A second bounding box
    :return: IoU
    """
    new_bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2] - 1, bbox1[1] + bbox1[3] - 1]
    new_bbox2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2] - 1, bbox2[1] + bbox2[3] - 1]
    return calculate_iou_two_points(new_bbox1, new_bbox2)
