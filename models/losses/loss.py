from misc.utils import TrainingParams
from models.losses.loss_utils import *
from models.losses.truncated_smoothap import TruncatedSmoothAP
from models.losses.pointinfonce import PointInfoNCELoss

def make_losses(params: TrainingParams):
    loss_fn_truncated = TruncatedSmoothAP(tau1=params.tau1, similarity=params.similarity,
                                positives_per_query=params.positives_per_query)
    loss_fn_infonce = PointInfoNCELoss(num_pos=params.num_pos, num_hn_samples=params.num_hn_samples,
                                      temperature=params.temperature, vdist=params.vdist,
                                      hdist=params.hdist, circular_horizontal=params.circular_horizontal)

    return (loss_fn_truncated, loss_fn_infonce)




