"""Flags used by uflow training and evaluation."""
from absl import flags


FLAGS = flags.FLAGS

# General flags.
flags.DEFINE_string('dataset', 'minecraft',
                    'Dataset to use, i.e. "minecraft" or "moving_object".')

flags.DEFINE_bool(
    'use_minecraft_camera_actions', True, 'If true, append camera actions and xy grid augmentation to flow layers input.')

flags.DEFINE_integer('num_epochs', 500,
                     'Number epochs for training.')

flags.DEFINE_integer('batch_size', 64,
                     'Batch size for training.')

flags.DEFINE_bool('continue_training', False, 'If true, continue training at previous checkpoint, otherwise start over.')

flags.DEFINE_string('device', 'auto',
                    'Device to use, i.e. "cpu" or "cuda:0", or "auto" to automatically select best GPU')

# loss flags
flags.DEFINE_float('weight_smooth1', 0.3, 'Weight for smoothness loss.')
flags.DEFINE_float('smoothness_edge_constant', 100.,
                   'Edge constant for smoothness loss.')
flags.DEFINE_float('weight_ssim', 0.1, 'Weight for SSIM loss.')