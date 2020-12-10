from collections import UserList
from enum import IntEnum, auto


# From https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#bounding-box-mode
class BoxMode(IntEnum):
    CXCYWH = 0
    """
    The bounding box is represented as [cx, cy, w, h], where xc and yc are the coordinates of the 
    bounding box centroid, and w and h are the width and height of the bounding box.
    """

    XYXY = auto()
    """
    The bounding box is represented as [xmin, ymin, xmax, ymax], where xmin and ymin are the 
    coordinates of the bounding box top-left corner, and xmax and ymax are the coordinates of the 
    bounding box bottom-right corner.
    """

    XXYY = auto()
    """
    The bounding box is represented as [xmin, xmax, ymin, ymax], where xmin and xmax are the 
    minimum and maximum of the x coordinates, and ymin and ymax are the minimum and maximum of the 
    y coordinates.
    """

    XYWH = auto()
    """
    The bounding box is represented as [x, y, w, h], where x and y are the coordinates of the 
    bounding box top-left corner, and w and h are the width and height of the bounding box.
    """

    def _generate_next_value_(self, start, count, last_values):
        """Generate consecutive automatic numbers starting from zero."""
        return count

    def new(self, data, relative=None, absolute=None):
        return BoundingBox(data, mode=self, relative=relative, absolute=absolute)


class BoundingBox(UserList):

    def new_like(self, data):
        return BoundingBox(data, mode=self.mode, relative=self.relative)

    def __init__(self, data, mode=BoxMode.XYXY, relative=None, absolute=None):
        assert isinstance(mode, BoxMode)
        assert len(data) == 4, 'Bounding box must have length of 4.'
        assert not (relative and absolute), 'Either relative or absolute can be passed as ' \
                                            'argument, not both.'

        super(BoundingBox, self).__init__(data)

        if relative is not None:
            self.relative = relative
        else:
            self.relative = absolute is not None and not absolute

        self.mode = mode

    def to(self, target_mode):
        assert isinstance(target_mode, BoxMode)

        if self.mode == target_mode:
            return self

        is_tgt_xyxy = target_mode == BoxMode.XYXY
        is_tgt_centroid = target_mode == BoxMode.CXCYWH
        is_tgt_xywh = target_mode == BoxMode.XYWH

        def create(data):
            return target_mode.new(data, relative=self.relative)

        # YOLO bounding box
        if self.mode == BoxMode.CXCYWH:
            xc, yc, w, h = self

            xmin = xc - w / 2
            ymin = yc - h / 2

            if is_tgt_xywh:
                return create([xmin, ymin, w, h])

            xmax = xmin + w
            ymax = ymin + h

            return create([xmin, ymin, xmax, ymax] if is_tgt_xyxy
                          else [xmin, xmax, ymin, ymax])

        # MS COCO bounding box
        if self.mode == BoxMode.XYWH:
            xmin, ymin, w, h = self

            if is_tgt_centroid:
                return create([xmin + w / 2, ymin + h / 2, w, h])

            xmax = xmin + w
            ymax = ymin + h

            return create([xmin, ymin, xmax, ymax] if is_tgt_xyxy
                          else [xmin, xmax, ymin, ymax])

        if self.mode == BoxMode.XYXY:
            xmin, ymin, xmax, ymax = self
        else:
            xmin, xmax, ymin, ymax = self  # self.mode == BoxMode.MIN_MAX

        if is_tgt_centroid:
            w = xmax - xmin
            h = ymax - ymin
            xc = xmin + w / 2
            yc = ymin + h / 2

            return create([xc, yc, w, h])

        if is_tgt_xywh:
            w = xmax - xmin
            h = ymax - ymin

            return create([xmin, ymin, w, h])

        return create([xmin, ymin, xmax, ymax] if is_tgt_xyxy
                      else [xmin, xmax, ymin, ymax])

    def resize(self, source_size, target_size):
        (src_w, src_h), (tgt_w, tgt_h) = source_size, target_size

        if self.relative or (src_w == tgt_w and src_h == tgt_h):
            return self

        ratio_w, ratio_h = tgt_w / src_w, tgt_h / src_h

        resized_bbox = self.to(BoxMode.XYXY)
        xmin, ymin, xmax, ymax = resized_bbox
        resized_bbox = resized_bbox.new_like(
            [xmin * ratio_w, ymin * ratio_h, xmax * ratio_w, ymax * ratio_h]
        )

        return resized_bbox.to(self.mode)

    def normalize(self, size):
        assert self.absolute

        w, h = size

        normalized_bbox = self.to(BoxMode.XYXY)
        xmin, ymin, xmax, ymax = normalized_bbox
        normalized_bbox = normalized_bbox.new_like([xmin / w, ymin / h, xmax / w, ymax / h])

        return normalized_bbox.to(self.mode)

    def denormalize(self, size):
        assert self.relative

        w, h = size

        denormalized_bbox = self.to(BoxMode.XYXY)
        xmin, ymin, xmax, ymax = denormalized_bbox
        denormalized_bbox = denormalized_bbox.new_like([xmin * w, ymin * h, xmax * w, ymax * h])

        return denormalized_bbox.to(self.mode)

    def area(self):
        return self.width * self.height

    @property
    def width(self):
        if self.mode == BoxMode.CXCYWH or self.mode == BoxMode.XYWH:
            return self[2]
        return (self[2] if self.mode == BoxMode.XYXY else self[1]) - self[0]

    @property
    def height(self):
        if self.mode == BoxMode.CXCYWH or self.mode == BoxMode.XYWH:
            return self[3]
        return self[3] - (self[1] if self.mode == BoxMode.XYXY else self[2])

    @property
    def absolute(self):
        return not self.relative

    def __eq__(self, other):
        return all([
            isinstance(other, BoundingBox),
            self.mode == other.mode,
            self.relative == other.relative,
            self.data == other.cache
        ])

    def __contains__(self, other):
        if not isinstance(other, BoundingBox):
            return False

        x0, y0, x1, y1 = self.to(BoxMode.XYXY)
        x2, y2, x3, y3 = other.to(BoxMode.XYXY)

        return all([
            x2 >= x0,
            y2 >= y0,
            x3 <= x1,
            y3 <= y1
        ])

    def __bool__(self):
        xmin, ymin, xmax, ymax = self.to(BoxMode.XYXY)
        return xmax > xmin and ymax > ymin

    def __repr__(self):
        return f'BoundingBox({self.data}, ' \
               f'mode={self.mode.name}, ' \
               f'relative={self.relative})'
