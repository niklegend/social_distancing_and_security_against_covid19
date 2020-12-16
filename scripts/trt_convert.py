import argparse
import os

import numpy as np
from masterthesis.utils import TimeIt
from tensorflow.python.compiler.tensorrt import trt_convert as trt

true_values = [
    'true',
    'y',
    'yes',
    '1',
]

false_values = [
    'false',
    'n',
    'no',
    '0',
]

bool_choices = set(true_values + false_values)


def str2bool(x):
    if x.lower() in true_values:
        return True
    if x.lower() in false_values:
        return False
    raise ValueError(f'Invalid boolean value: \'{x}\'')


precision_mode_to_dtpye = {
    trt.TrtPrecisionMode.INT8: np.uint8,
    trt.TrtPrecisionMode.FP16: np.float16,
    trt.TrtPrecisionMode.FP32: np.float32
}


def input_fn(shape, dtype, num_calibration_steps=1, gen_fn=None):
    if gen_fn is None:
        gen_fn = lambda x: np.random.randint(0, 256, x)

    def func():
        for _ in range(num_calibration_steps):
            inp = gen_fn(shape).astype(dtype=dtype)
            yield (inp,)

    return func


def trt_convert(
        input_saved_model_dir,
        output_saved_model_dir,
        conversion_params=None,
        input_shape=None,
        num_calibration_steps=1,
        input_type=None
):
    if os.path.exists(output_saved_model_dir):
        if not os.path.isdir(output_saved_model_dir):
            raise RuntimeError(f'\'{output_saved_model_dir}\' is not a directory.')
        if os.listdir(output_saved_model_dir):
            raise RuntimeError(f'\'{output_saved_model_dir}\' directory is not empty.')

    if not conversion_params:
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS

    if input_shape:
        input_shape = 1, *input_shape

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params
    )

    calibration_input_fn = None

    if conversion_params.precision_mode == trt.TrtPrecisionMode.INT8:
        calibration_input_fn = input_fn(
            input_shape,
            np.uint8,
            num_calibration_steps
        )

    with TimeIt(f'Converted files have been saved to {output_saved_model_dir}'):
        converter.convert(calibration_input_fn=calibration_input_fn)

        if input_shape:
            converter.build(
                input_fn=input_fn(
                    input_shape,
                    precision_mode_to_dtpye[input_type],
                    gen_fn=np.zeros
                )
            )

        os.makedirs(output_saved_model_dir, exist_ok=True)
        converter.save(output_saved_model_dir)


def main(args):
    conversion_params = trt.TrtConversionParams(
        max_workspace_size_bytes=args.max_workspace_size_bytes,
        precision_mode=args.precision_mode,
        minimum_segment_size=args.minimum_segment_size,
        is_dynamic_op=args.is_dynamic_op,
        maximum_cached_engines=args.maximum_cached_engines,
        use_calibration=args.use_calibration,
        max_batch_size=args.max_batch_size,
        allow_build_at_runtime=args.allow_build_at_runtime
    )

    trt_convert(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        conversion_params=conversion_params,
        input_shape=args.input_shape,
        num_calibration_steps=args.num_calibration_steps,
        input_type=args.input_type
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='An offline converter for TF-TRT transformation for TF 2.0 SavedModels.\n\n'
                    'Currently this is not available on Windows platform.\n\n'
                    'Note that in V2, is_dynamic_op=False is not supported, meaning TRT engines '
                    'will be built only when the corresponding TRTEngineOp is executed. But we '
                    'still provide a way to avoid the cost of building TRT engines during '
                    'inference (see more below).'
    )

    parser.add_argument(
        '--input-saved-model-dir',
        required=True,
        help='the directory to load the SavedModel which contains the input graph to transforms.'
    )

    parser.add_argument(
        '--output-saved-model-dir',
        required=True,
        help='directory to save the converted SavedModel.'
    )

    ########################################
    # Conversion params
    ########################################

    parser.add_argument(
        '--max-workspace-size-bytes',
        type=int,
        default=trt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
        help='the maximum GPU temporary memory which the TRT engine can use at execution time. '
             'This corresponds to the \'workspaceSize\' parameter of '
             'nvinfer1::IBuilder::setMaxWorkspaceSize().'
    )

    parser.add_argument(
        '--precision-mode',
        default=trt.TrtPrecisionMode.FP32,
        choices=trt.TrtPrecisionMode.supported_precision_modes(),
        help='one the strings in {supported_precision_modes}.'
    )

    parser.add_argument(
        '--minimum-segment-size',
        type=int,
        default=3,
        help='the minimum number of nodes required for a subgraph '
             'to be replaced by TRTEngineOp.'
    )

    parser.add_argument(
        '--is-dynamic-op',
        type=str2bool,
        choices=bool_choices,
        default=true_values[0],
        help='whether to generate dynamic TRT ops which will build the TRT network and engine at '
             'run time. i.e. Since TensorRT version < 6.0 does not support dynamic dimensions '
             'other than the batch dimension, when the TensorFlow graph has a non-batch dimension '
             'of dynamic size, we would need to enable this option. This   option should be set to '
             'True in TF 2.0.'
    )

    parser.add_argument(
        '--maximum-cached-engines',
        type=int,
        default=1,
        help='max number of cached TRT engines for dynamic TRT ops. Created TRT engines for a '
             'dynamic dimension are cached. This is the maximum number of engines that can be '
             'cached. If the number of cached engines is already at max but none of them supports '
             'the input shapes, the TRTEngineOp will fall back to run the original TF subgraph '
             'that corresponds to the TRTEngineOp.'
    )

    parser.add_argument(
        '--use-calibration',
        type=str2bool,
        choices=bool_choices,
        default=true_values[0],
        help='this argument is ignored if precision_mode is not INT8. If set to True, a '
             'calibration graph will be created to calibrate the missing ranges. The calibration '
             'graph must be converted to an inference graph by running calibration with '
             'calibrate(). If set to False, quantization nodes will be expected for every tensor '
             'in the graph (excluding those which will be fused). If a range is missing, an error '
             'will occur. Please note that accuracy may be negatively affected if there is a '
             'mismatch between which tensors TRT quantizes and which tensors were trained with '
             'fake quantization.'
    )

    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='max size for the input batch. This parameter is only effective when '
             'is_dynamic_op=False which is not supported in TF 2.0.'
    )

    parser.add_argument(
        '--allow-build-at-runtime',
        type=str2bool,
        choices=bool_choices,
        default=true_values[0],
        help='whether to build TensorRT engines during runtime. If no TensorRT engine can be found '
             'in cache that can handle the given inputs during runtime, then a new TensorRT engine '
             'is built at runtime if allow_build_at_runtime=True, and otherwise native TF is used. '
             'This argument is only effective if is_dynamic_op=True.'
    )

    parser.add_argument(
        '--num_calibration_steps',
        type=int,
        default=1,
        help='number of calibration steps to perform. This argument is ignored if INT8 calibration '
             'is not needed.'
    )

    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        help='shape of the input data yielded by the generator function, which will be used '
             'to execute the converted signature to generate TRT engines. Input shape is expected '
             'as (Height, Width, Channels).'
    )

    parser.add_argument(
        '--input-type',
        type=str,
        choices=trt.TrtPrecisionMode.supported_precision_modes(),
        help='type of the input data yielded by the generator function, which will be used to '
             'execute the converted signature to generate TRT engines.'
    )

    args = parser.parse_args()

    if args.precision_mode == trt.TrtPrecisionMode.INT8 and args.input_shape is None:
        parser.error('INT8 precision mode requires --input-shape')

    if args.input_shape is not None and args.input_type is None:
        parser.error('Should specify --input-type when --input-shape is provided')

    main(args)
