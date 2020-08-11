import uflow_model
import uflow_utils

class UFlow:
    def __init__(self, num_levels = 5, use_cost_volume=True):
        self._pyramid = uflow_model.PWCFeaturePyramid(num_levels=num_levels).cuda()
        self._flow_model = uflow_model.PWCFlow(num_levels = num_levels, num_channels_upsampled_context=32, use_cost_volume=use_cost_volume, use_feature_warp=True).cuda()

    def compute_flow(self, img1, img2):
        fp1 = self._pyramid(img1)
        fp2 = self._pyramid(img2)
        flow = self._flow_model(fp1, fp2)
        return flow, fp1, fp2

    def compute_loss(self, img1, img2, features1, features2, flows):
        f1 = [img1] + features1
        f2 = [img2] + features2

        warps = [uflow_utils.flow_to_warp(f) for f in flows]
        warped_f2 = [uflow_utils.resample(f, w) for (f, w) in zip(f2, warps)]

        loss = uflow_utils.compute_all_loss(f1, warped_f2, flows)
        return loss
