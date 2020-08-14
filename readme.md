# UFlow implementation in PyTorch

This is an unofficial implementation of https://github.com/google-research/google-research/tree/master/uflow

Currently in very early stage, only basic loss functions implemented.

For training with minecraft data, the ep1_pickle_doc.pkl must be placed into dataset/UFlow_data folder (see dataset.py)

For training, use main.py, for showing some test results use show_results.py

Torch checkpoint is stored at save/checkpoint.pt

Flags:

Set use_minecraft_camera_actions=True/False inside main.py and show_result.py to enable/disable camera actions. If
disabled, the channels still will be present but set to zero. Remember to set the flag in both files!

Set continue_training=True/False to continue training previous checkpoint and false to begin a new training. Note that 
the checkpoint will be overridden.

Inside uflow_utils.compute_all_loss some loss weights (smoothness_weight and ssim_weight) can be adjusted.
