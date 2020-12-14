import argparse
import os
import shutil

from masterthesis.utils import TimeIt

THIS_FILE_DIR = os.path.dirname(__file__)

directories = [
    'annotations',
    'data',
    'exported-models',
    'models',
    'pre-trained-models'
]


def create_workspace(workspace_root, scripts_path):
    if os.path.exists(workspace_root):
        print(f'Could not create workspace at {workspace_root}.')
        print('Directory already exists.')
        exit(1)

    os.makedirs(os.path.dirname(workspace_root), exist_ok=True)

    with TimeIt(f'Wowkspace created at {workspace_root}'):
        shutil.copytree(scripts_path, workspace_root)
        for d in directories:
            os.mkdir(os.path.join(workspace_root, d))


def main(args):
    workspace_root = args.workspace_root

    if args.tf1:
        scripts_dir = '.TF1_SCRIPTS'
    else:
        scripts_dir = '.TF2_SCRIPTS'

    scripts_path = os.path.join(THIS_FILE_DIR, scripts_dir)

    create_workspace(workspace_root, scripts_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('workspace_root', help='Directory where the workspace will be created.')

    tf_group = parser.add_mutually_exclusive_group(required=True)
    tf_group.add_argument(
        '--tf1',
        action='store_true',
        help='Use TensorFlow Object Detection API 1.13.0 scripts.'
    )
    tf_group.add_argument(
        '--tf2',
        action='store_true',
        help='Use TensorFlow Object Detection API 2.3.0 scripts.'
    )

    args = parser.parse_args()
    main(args)
