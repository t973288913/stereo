"""The loss aggregator."""

from . import losses
from utils import Odict
from utils import is_dict, get_attr_from, get_valid_args, is_tensor, get_ddp_module, get_msg_mgr
import pdb
import torch
class LossAggregator_mul():
    """The loss aggregator.

    This class is used to aggregate the losses.
    For example, if you have two losses, one is triplet loss, the other is cross entropy loss,
    you can aggregate them as follows:
    loss_num = tripley_loss + cross_entropy_loss

    Attributes:
        losses: A dict of losses.
    """

    def __init__(self, loss_cfg) -> None:
        """
        Initialize the loss aggregator.

        Args:
            loss_cfg: Config of losses. List for multiple losses.
        """
        # pdb.set_trace()
        
        # self.losses = {loss_cfg['log_prefix']: self._build_loss_(loss_cfg)} if is_dict(loss_cfg) \
        #     else {cfg['log_prefix']: 
        #           self._build_loss_(cfg) for cfg in loss_cfg
        #           }
        self.losses = {cfg['log_prefix']: self._build_loss_(cfg) for cfg in loss_cfg }
        # pdb.set_trace()
        
                

        

    def _build_loss_(self, loss_cfg):
        """Build the losses from loss_cfg.

        Args:
            loss_cfg: Config of loss.
        """
        # pdb.set_trace()

        Loss = get_attr_from([losses], loss_cfg['type'])
        valid_loss_arg = get_valid_args(
            Loss, loss_cfg, ['type', 'gather_and_scale'])
        loss = get_ddp_module(Loss(**valid_loss_arg))
        
        return loss

    def __call__(self, training_disp):
        """Compute the sum of all losses.

        The input is a dict of features. The key is the name of loss and the value is the feature and label. If the key not in
        built losses and the value is torch.Tensor, then it is the computed loss to be added loss_sum.

        Args:
            training_disp: A dict of features. The same as the output["training_feat"] of the model.
            LossAggregator_mul training_disp:   dict_keys(['disp', 'seg'])
            self.losses:   dict_keys(['disp', 'seg'])
        """
        loss_sum = 0.
        loss_info = Odict()

        # print("LossAggregator_mul training_disp:  ",training_disp.keys() )
        # print("LossAggregator_mul self.losses:  ",self.losses.keys() )
        # pdb.set_trace()
        # pdb.set_trace()

        for k, v in training_disp.items():
            if k in self.losses:
                
                loss_func = self.losses[k]
                # pdb.set_trace()
                                
                loss, info = loss_func(**v)
                for name, value in info.items():
                    loss_info[f'scalar/train/loss_{k}'] = value
                loss = loss.mean() * loss_func.loss_term_weight
                loss_sum += loss
            else:
                if isinstance(v, dict):
                    raise ValueError(
                        "The key %s in -trainng_disp- should be stated as the log_prefix of a certain loss defined in your loss_cfg." % v
                    )
                elif is_tensor(v):
                    _ = v.mean()
                    loss_info['scalar/%s' % k] = _
                    loss_sum += _
                    get_msg_mgr().log_debug(
                        "Please check whether %s needed in training." % k)
                else:
                    raise ValueError(
                        "Error type for -training_disp-, supported: A feature dict or loss tensor.")
        # loss_info['scalar/train/loss_sum'] = loss_sum
        return loss_sum, loss_info
