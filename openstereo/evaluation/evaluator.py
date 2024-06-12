from functools import partial
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from evaluation.metric import *
import pdb


METRICS = {
    'miou' : compute_mIoU,
    # EPE metric (Average Endpoint Error)
    'epe': epe_metric_per_image,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'd1_all': d1_metric_per_image,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    'bad_1': partial(bad_metric_per_image, threshold=1),
    'bad_2': partial(bad_metric_per_image, threshold=2),
    'bad_3': partial(bad_metric_per_image, threshold=3),
}


class OpenStereoEvaluator:
    def __init__(self, metrics=None):
        # Set default metrics if none are given
        if metrics is None:
            metrics = ['epe', 'd1_all']
        self.metrics = metrics

    def __call__(self, data):
        # Extract input data
       
        disp_est = data['disp_est']
        disp_gt = data['disp_gt']
        mask = data['mask']
        if 'seg_labels' in data:
            seg_gt = data['seg_labels']
            seg_est = data['seg_pred']
            num_class = seg_gt.size(3)  #numclass等于seg_gt 的第四维度的个数
        
        res = {}
        # pdb.set_trace()
        # Loop through the specified metrics and compute results
        for m in self.metrics:
            # Check if the metric is valid
            if m not in METRICS:
                raise ValueError("Unknown metric: {}".format(m))
            else:
                # Get the appropriate metric function based on use_np
                # pdb.set_trace()
                metric_func = METRICS[m]
                # import pdb;pdb.set_trace()                
                if  (m == "miou" or m == "f1") and 'seg_labels' in data:
                    # Compute the metric and store the result in the dictionary
                    res[m] = metric_func(seg_est,seg_gt,num_class)
                elif m == 'epe' or m == "d1_all" or m == "bad_1" or m == "bad_2" or m == "bad_3":
                    # pdb.set_trace()
                    res[m] = metric_func(disp_est, disp_gt, mask)
        # pdb.set_trace()
        return res
    

   

# from functools import partial
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from evaluation.metric import *

# METRICS = {
#     # EPE metric (Average Endpoint Error)
#     'epe': epe_metric_per_image,
#     # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
#     'd1_all': d1_metric_per_image,
#     # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
#     'bad_1': partial(bad_metric_per_image, threshold=1),
#     'bad_2': partial(bad_metric_per_image, threshold=2),
#     'bad_3': partial(bad_metric_per_image, threshold=3),
# }


# class OpenStereoEvaluator:
#     def __init__(self, metrics=None):
#         # Set default metrics if none are given
#         if metrics is None:
#             metrics = ['epe', 'd1_all']
#         self.metrics = metrics

#     def __call__(self, data):
#         # Extract input data
#         disp_est = data['disp_est']
#         disp_gt = data['disp_gt']
#         mask = data['mask']
#         res = {}
#         # Loop through the specified metrics and compute results
#         for m in self.metrics:
#             # Check if the metric is valid
#             if m not in METRICS:
#                 raise ValueError("Unknown metric: {}".format(m))
#             else:
#                 # Get the appropriate metric function based on use_np
#                 metric_func = METRICS[m]

#                 # Compute the metric and store the result in the dictionary
#                 res[m] = metric_func(disp_est, disp_gt, mask)
#         return res
   

