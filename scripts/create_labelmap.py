import argparse
import os
import re

from masterthesis.utils import quote, TimeIt


def to_display_name(class_name):
    return re.sub('[-_ ]+', ' ', class_name).capitalize()


def create_item_str(class_id, class_name, display_name=None):
    def append_line(key, value, indent=2):
        return ''.join([*([' '] * indent), key, ': ', value])

    if not display_name:
        display_name = to_display_name(class_name)

    return '\n'.join([
        'item {',
        append_line('name', quote(class_name)),
        append_line('display_name', quote(display_name)),
        append_line('id', str(class_id)),
        '}'
    ])


def create_labelmap_str(class_names, display_names=None, default_class=None):
    if display_names is None:
        display_names = [to_display_name(class_name) for class_name in class_names]
    else:
        diff = abs(len(display_names) - len(class_names))
        if diff != 0:
            print(f'The number of display names ({len(display_names)}) must be the same as the '
                  f'number of classes ({len(class_names)}).')
            exit(diff)

    labelmap_str = []
    if default_class:
        labelmap_str.append(create_item_str(0, default_class))
    for class_id, (class_name, display_name) in enumerate(zip(class_names, display_names), start=1):
        labelmap_str.append(create_item_str(class_id, class_name, display_name))

    return '\n'.join(labelmap_str)


def create_labelmap(output_path, class_names, display_names=None, default_class=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with TimeIt(f'Successfully created label map file {output_path}'):
        with open(output_path, 'w') as f:
            f.write(create_labelmap_str(class_names, display_names, default_class))
            f.write('\n')


def main(args):
    class_names = args.class_names
    display_names = args.display_names
    default_class = args.default_class

    output_path = args.output_path

    create_labelmap(output_path, class_names, display_names, default_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('class_names', metavar='class_name', nargs='+')
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('--display-names', nargs='+')
    parser.add_argument('--default-class')

    main(parser.parse_args())
