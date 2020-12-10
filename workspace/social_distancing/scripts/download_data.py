import argparse
import os
from argparse import Namespace

from masterthesis.data import download_file


def gdrive_url(file_id):
    return 'https://drive.google.com/uc?id=%s' % file_id


widerperson = Namespace(
    url=gdrive_url('1I7OjhaomWqd8Quf7o5suwLloRlY0THbp'),
    name='WiderPerson.zip',
    download_type='gdrive'
)


def download_data(root, download_root=None, remove_finished=False):
    if download_root is None:
        download_root = root

    os.makedirs(root, exist_ok=True)
    os.makedirs(download_root, exist_ok=True)

    from_path = os.path.join(download_root, widerperson.name)
    to_path = os.path.join(root, os.path.splitext(widerperson.name)[0])

    download_file(
        widerperson.url,
        widerperson.download_type,
        from_path=from_path,
        to_path=to_path,
        remove_finished=remove_finished
    )


def main(args):
    download_data(
        root=args.root,
        download_root=args.download_root,
        remove_finished=args.remove_finished
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', required=True)
    parser.add_argument('--download-root')
    parser.add_argument('--remove-finished', action='store_true')

    args = parser.parse_args()

    main(args)
