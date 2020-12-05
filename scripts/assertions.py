def is_bounding_box_within_image_size(xmin, ymin, xmax, ymax, width, height):
    return xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height


def is_valid_bounding_box(xmin, ymin, xmax, ymax):
    return xmax > xmin and ymax > ymin
