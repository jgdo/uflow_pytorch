import torch
import uflow_model

# just check if network forwarding works

img1 = torch.rand((1, 3, 256, 256))
img2 = torch.rand_like(img1)

pyramid = uflow_model.PWCFeaturePyramid()

fp1 = pyramid(img1)
fp2 = pyramid(img2)

flow_model = uflow_model.PWCFlow(num_channels_upsampled_context=0, use_cost_volume=False, use_feature_warp=False)

flow = flow_model(fp1, fp2)

print(flow)

