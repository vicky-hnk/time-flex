import sys
from src.experiments.main import exec_shell
# import models
from src.models.autoformer import Autoformer
from src.models.crossformer import CrossFormer
from src.models.cyclenet import CycleNet
from src.models.d_linear import DLinearModel
from src.models.fedformer import FEDformer
from src.models.i_transfomer import Itransformer
from src.models.n_linear import NLinear
from src.models.patch_tst import PatchTST
from src.models.r_linear import RLinear
from src.models.sparse_tsf import SparseTSF
from src.models.tide import Tide
from src.models.time_flex import TimeFlex
from src.models.time_flex_jpp import TimeFlexJPP
from src.models.time_mixer import TimeMixer
from src.models.vcformer import VCFormer
from src.models.wave_net import WaveNet


def get_model(selected_model):
    """Map model name to corresponding model class dynamically."""
    selected_model = selected_model.lower()
    model_dict = {
        'autoformer': Autoformer,
        'crossformer': CrossFormer,
        'cyclenet': CycleNet,
        'dlinear': DLinearModel,
        'fedformer': FEDformer,
        'itransformer': Itransformer,
        'nlinear': NLinear,
        'patchtst': PatchTST,
        'rlinear': RLinear,
        'sparse_tsf': SparseTSF,
        'tide': Tide,
        'timeflex': TimeFlex,
        'timeflexjpp': TimeFlexJPP,
        'timemixer': TimeMixer,
        'vcformer': VCFormer,
        'wavemask': DLinearModel,
        'wavenet': WaveNet
    }
    if selected_model in model_dict:
        return model_dict[selected_model]
    else:
        raise ValueError(f"Unknown model: {selected_model}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python exec_entry.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    model_class = get_model(model_name)
    exec_shell(model_type=model_class)
