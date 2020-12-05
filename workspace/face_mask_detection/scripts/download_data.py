import argparse
import os
from argparse import Namespace

from kaggle.api import KaggleApi

import gdown
from masterthesis.utils import TimeIt
from torchvision.datasets.utils import download_url, extract_archive


def gdrive_url(file_id):
    return f'https://drive.google.com/uc?id={file_id}'


kaggle = Namespace(
    dataset='ivandanilovich/medical-masks-dataset-images-tfrecords',
    name='medical-masks-dataset-images-tfrecords.zip',
    download_type='url'
)

mafa_files = [
    Namespace(
        url=gdrive_url('17bRIiaGyrKLEDQOV2RlqbPQ9TyCZxq9k'),
        name='train-images.zip',
        download_type='gdrive'
    ),
    Namespace(
        url=gdrive_url('1Fu1C1O8ok-Z7r8XSWoTb9yB_2_5w6BDt'),
        name='MAFA-Label-Train.zip',
        download_type='gdrive'
    ),
    Namespace(
        url=gdrive_url('1jJHdmmscqxvNQ2dxKUrLaHqW3w1Yo_9S'),
        name='test-images.zip',
        download_type='gdrive'
    ),
    Namespace(
        url=gdrive_url('1uN0a4P0wAFwJLid_r7VHFs0KUcizIRGN'),
        name='MAFA-Label-Test.zip',
        download_type='gdrive'
    )
]

fddb_files = [
    Namespace(
        url='http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz',
        name='originalPics.tar.gz',
        download_type='url'
    ),
    Namespace(
        url='http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz',
        name='FDDB-folds.tar.gz',
        download_type='url'
    )
]

widerface_files = [
    Namespace(
        url=gdrive_url('0B6eKvaijfFUDQUUwd21EckhUbWs'),
        name='WIDER_train.zip',
        download_type='gdrive'
    ),
    Namespace(
        url=gdrive_url('0B6eKvaijfFUDd3dIRmpvSk8tLUk'),
        name='WIDER_val.zip',
        download_type='gdrive'
    ),
    Namespace(
        url=gdrive_url('0B6eKvaijfFUDbW4tdGpaYjgzZkU'),
        name='WIDER_test.zip',
        download_type='gdrive'
    ),
    Namespace(
        url='http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip',
        name='wider_face_split.zip',
        download_type='url'
    )
]


def download_kaggle(root, download_root, remove_finished=False):
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(download_root):
        os.makedirs(download_root)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        kaggle.dataset,
        path=download_root,
        quiet=False
    )

    extract_archive(
        from_path=os.path.join(download_root, kaggle.name),
        to_path=root,
        remove_finished=remove_finished
    )


def download_files(files, root, download_root, create_extract_dir=False, remove_finished=False):
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(download_root):
        os.makedirs(download_root)

    for file in files:
        url = file.url
        from_path = os.path.join(download_root, file.name)
        if create_extract_dir:
            to_path = os.path.join(root, os.path.splitext(file.name)[0])
        else:
            to_path = root
        download_type = file.download_type

        with TimeIt('Download complete.'):
            if download_type == 'gdrive':
                gdown.download(url, from_path, quiet=False)
            elif download_type == 'url':
                dirnaname, basename = os.path.split(from_path)
                download_url(url, root=dirnaname, filename=basename)
            else:
                raise ValueError(f'Invalid \'download_type\': {download_type}')

        with TimeIt('Extraction complete.'):
            print(f'Extracting {from_path} to {to_path}')
            extract_archive(
                from_path=from_path,
                to_path=to_path,
                remove_finished=remove_finished
            )


def download_data(root, download_root=None, remove_finished=False):
    if download_root is None:
        download_root = root

    download_kaggle(
        root=os.path.join(root, 'kaggle'),
        download_root=os.path.join(download_root, 'kaggle'),
        remove_finished=remove_finished
    )

    print()

    download_files(
        mafa_files,
        download_root=os.path.join(download_root, 'MAFA'),
        root=os.path.join(root, 'MAFA'),
        remove_finished=remove_finished,
        create_extract_dir=True
    )

    print()

    download_files(
        fddb_files,
        download_root=os.path.join(download_root, 'FDDB'),
        root=os.path.join(root, 'FDDB'),
        remove_finished=remove_finished
    )

    print()

    download_files(
        widerface_files,
        download_root=os.path.join(download_root, 'WiderFace'),
        root=os.path.join(root, 'WiderFace'),
        remove_finished=remove_finished
    )


def main(args):
    download_data(
        root=args.root,
        download_root=args.download_root,
        remove_finished=args.archives_root
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', required=True)
    parser.add_argument('--download-root')
    parser.add_argument('--remove-finished', action='store_true')

    args = parser.parse_args()

    main(args)
