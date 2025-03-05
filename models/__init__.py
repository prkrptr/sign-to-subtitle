from .gt_align_model import *
from .gt_align_model_neg_rel import *

model_dict = {
    'gt_align_invtransformer': GtInvAlignTransformer,
    'gt_align_invtransformer_neg_rel': GtInvAlignTransformer_neg_rel,
}

__all__ = ['model_dict']
