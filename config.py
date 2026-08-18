import os
import time
from pathlib import Path

from easydict import EasyDict as edict


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "DeepFish"

__C = edict()
cfg = __C


__C.SEED = 3035
__C.DATASET = "DeepFish"
__C.NAME = "train"
__C.ABLATION_PROFILE = "full"
__C.encoder = "VGG16_FPN"
__C.BACKBONE_PRETRAINED = False
__C.RESUME = False
__C.RESUME_PATH = ""
__C.PRE_TRAIN_COUNTER = os.environ.get("DVSD_PRETRAIN_COUNTER", "")

__C.TRAIN_ROOT = os.environ.get("DEEPFISH_TRAIN_ROOT", str(DEFAULT_DATA_ROOT / "train"))
__C.VAL_ROOT = os.environ.get("DEEPFISH_VAL_ROOT", str(DEFAULT_DATA_ROOT / "val"))
__C.INPUT_SIZE = 512
__C.MAX_POINTS = 512
__C.MIN_CROP_SCALE = 0.75
__C.TRAIN_BATCH_SIZE = 2
__C.VAL_BATCH_SIZE = 4
__C.NUM_WORKERS = 4

__C.GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = __C.GPU_ID

__C.cross_attn_embed_dim = 256
__C.cross_attn_num_heads = 4
__C.mlp_ratio = 4
__C.cross_attn_depth = 2
__C.DCFA_STAGES = 3

__C.FEATURE_DIM = 256
__C.USE_DCFA = True
__C.USE_DCN = True
__C.USE_FSG = False
__C.USE_UPHY_ENHANCE = True
__C.USE_ENHANCE_MASK_SUPERVISION = True
__C.UPHY_ENHANCE_CHANNELS = 32
__C.ENHANCE_LR_MULT = 5.0
__C.ENHANCE_WARMUP_EPOCHS = 5
__C.USE_SHARE_BRANCH = True

__C.LAMBDA_GLOBAL = 1.0
__C.LAMBDA_SHARE = 0.5
__C.LAMBDA_COUNT = 0.001
__C.LAMBDA_ENH_COLOR = 0.01
__C.LAMBDA_ENH_EDGE = 0.05
__C.LAMBDA_ENH_TV = 0.001

__C.LR_Base = 1e-5
__C.WEIGHT_DECAY = 1e-6
__C.MAX_EPOCH = 100
__C.VAL_INTERVAL = 1
__C.START_VAL = 0
__C.SAVE_INTERVAL = 1
__C.PRINT_FREQ = 20

__C.ENCODER_CHOICES = ("VGG16_FPN",)
__C.ABLATION_PROFILES = (
    "global_only",
    "dcfa_baseline",
    "full",
)


def apply_ablation_profile(config, profile):
    profile = profile.lower()
    if profile not in config.ABLATION_PROFILES:
        raise ValueError(f"unsupported profile: {profile}")

    config.ABLATION_PROFILE = profile
    config.USE_DCFA = True
    config.USE_DCN = False
    config.USE_FSG = False
    config.USE_UPHY_ENHANCE = False
    config.USE_ENHANCE_MASK_SUPERVISION = True
    config.ENHANCE_LR_MULT = 5.0
    config.ENHANCE_WARMUP_EPOCHS = 5
    config.USE_SHARE_BRANCH = True
    config.LAMBDA_GLOBAL = 1.0
    config.LAMBDA_SHARE = 0.5
    config.LAMBDA_COUNT = 0.001
    config.LAMBDA_ENH_COLOR = 0.01
    config.LAMBDA_ENH_EDGE = 0.05
    config.LAMBDA_ENH_TV = 0.001

    if profile == "global_only":
        config.USE_DCFA = False
        config.USE_SHARE_BRANCH = False
        config.LAMBDA_SHARE = 0.0
    elif profile == "dcfa_baseline":
        pass
    elif profile == "full":
        config.USE_DCN = True
        config.USE_UPHY_ENHANCE = True


def _bool_tag(enabled):
    return "1" if enabled else "0"


def _build_variant_name(config):
    return (
        f"{config.encoder}"
        f"_p-{config.ABLATION_PROFILE}"
        f"_bbpre{_bool_tag(getattr(config, 'BACKBONE_PRETRAINED', False))}"
        f"_dcfa{_bool_tag(config.USE_DCFA)}"
        f"_dcn{_bool_tag(config.USE_DCN)}"
        f"_fsg{_bool_tag(config.USE_FSG)}"
        f"_enh{_bool_tag(config.USE_UPHY_ENHANCE)}"
        f"_enhmask{_bool_tag(config.USE_ENHANCE_MASK_SUPERVISION)}"
        f"_share{_bool_tag(config.USE_SHARE_BRANCH and config.LAMBDA_SHARE > 0)}"
    )


def refresh_config(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
    now = time.strftime("%m-%d_%H-%M", time.localtime())
    variant = _build_variant_name(config)
    config.EXP_NAME = f"{now}_{config.DATASET}_{config.LR_Base}_{variant}_{config.NAME}"
    config.VAL_VIS_PATH = "./exp/" + config.DATASET + "_val"
    config.EXP_PATH = os.path.join("./exp", config.DATASET)
    os.makedirs(config.EXP_PATH, exist_ok=True)


refresh_config(__C)
