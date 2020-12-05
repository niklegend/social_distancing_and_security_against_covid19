from .tokitticonverter import Category

__all__ = [
    'Category'
]

logs = None


def log(image_path, width, height, bbox):
    global logs
    if logs is not None:
        logs.append({
            'image_path': image_path,
            'size': {
                'width': width,
                'height': height,
            },
            'bbox': bbox
        })
