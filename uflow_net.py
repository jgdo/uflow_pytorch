import uflow_model
import uflow_utils

class UFlow:
    def __init__(self, num_levels = 5):
        self._pyramid = uflow_model.PWCFeaturePyramid(num_levels=num_levels).cuda()
        self._flow_model = uflow_model.PWCFlow(num_levels = num_levels, num_channels_upsampled_context=32, use_cost_volume=True, use_feature_warp=True).cuda()

    def compute_flow(self, img1, img2):
        fp1 = self._pyramid(img1)
        fp2 = self._pyramid(img2)
        flow = self._flow_model(fp1, fp2)
        return flow

    def compute_loss(self, img1, img2, flows):
        flow = uflow_utils.upsample(flows[0], is_flow=True)
        flow = uflow_utils.upsample(flow, is_flow=True)
        warps = uflow_utils.flow_to_warp(flow)
        warped_images2 = uflow_utils.resample(img2, warps)

        loss = uflow_utils.compute_loss(img1, img2, flow, warped_images2)#
        return loss