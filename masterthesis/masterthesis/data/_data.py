import os

import gdown
from torchvision.datasets.utils import download_url, extract_archive
from masterthesis.utils import TimeIt


def download_file(url, download_type, from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with TimeIt():
        if download_type == 'gdrive':
            gdown.download(url, from_path, quiet=False)
        elif download_type == 'url':
            root, filename = os.path.split(from_path)
            download_url(url, root=root, filename=filename)
        else:
            raise ValueError(f'Invalid \'download_type\': {download_type}')

    with TimeIt():
        print(f'Extracting {from_path} to {to_path}')
        extract_archive(
            from_path=from_path,
            to_path=to_path,
            remove_finished=remove_finished
        )
