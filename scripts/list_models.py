import argparse


def list_models(model_names, model_url_fn, model_url_flag):
    print('List of available models:')
    print()

    for model_name in model_names:
        print(f' -  {model_name}')

        if model_url_flag:
            print(f'     -  Model URL: {model_url_fn(model_name)}')
            print()


def main(args):
    model_url_flag = args.model_url

    if args.tf1:
        from download_model import tf1_model_names, tf1_model_url
        list_models(tf1_model_names, tf1_model_url, model_url_flag)
    else:
        from download_model import tf2_model_names, tf2_model_url
        list_models(tf2_model_names, tf2_model_url, model_url_flag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-u', '--model-url',
        action='store_true',
        help='Show pre-trained model download URL'
    )

    tf_group = parser.add_mutually_exclusive_group(required=True)
    tf_group.add_argument(
        '--tf1',
        action='store_true',
        help='Show TensorFlow Object Detection API 1.13.0 pre-trained models.'
    )
    tf_group.add_argument(
        '--tf2',
        action='store_true',
        help='Show TensorFlow Object Detection API 2.3.0 pre-trained models.'
    )

    args = parser.parse_args()

    main(args)
