from fvcore.common.config import CfgNode

_C = CfgNode()
_C.PARSIMONY_PROBLEM = True
_C.AMP = False

# GflowNet
_C.GFN = CfgNode()
_C.GFN.LOSS_TYPE = 'TB'
_C.GFN.BACKWARD_MODELING = 'UNIFORM'
_C.GFN.NORMALIZE_LIKELIHOOD = False
_C.GFN.LIKELIHOOD_FLOAT16 = False
_C.GFN.CONDITION_ON_SCALE = False
_C.GFN.SCALES_SET = [2.0, 1.8, 1.5, 1.3, 1.0]
_C.GFN.SCALES_SAMPLING_MU = 2.0
_C.GFN.SCALES_SAMPLING_SIGMA = 0.5

_C.GFN.TRAINING_DATA_LOADER = CfgNode()
_C.GFN.TRAINING_DATA_LOADER.GFN_BATCH_SIZE = 32
_C.GFN.TRAINING_DATA_LOADER.BEST_STATE_BATCH_SIZE = 16
_C.GFN.TRAINING_DATA_LOADER.GFN_FIXED_SHAPE_BATCH_SIZE = 0
_C.GFN.TRAINING_DATA_LOADER.RANDOM_BATCH_SIZE = 16
_C.GFN.TRAINING_DATA_LOADER.BEST_TREES_BUFFER_SIZE = 100
_C.GFN.TRAINING_DATA_LOADER.MINI_BATCH_SPLITS = 1
_C.GFN.TRAINING_DATA_LOADER.NUM_WORKERS = 8
_C.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM = 100
_C.GFN.TRAINING_DATA_LOADER.FREQ_UPDATE_MODEL_WEIGHTS = 5
_C.GFN.TRAINING_DATA_LOADER.RANDOM_ACTION_PROB = 0.001
_C.GFN.TRAINING_DATA_LOADER.STEPS_PER_EPOCH = 100
_C.GFN.TRAINING_DATA_LOADER.PERTURB_BUFFERED_TREE = False
_C.GFN.TRAINING_DATA_LOADER.PIN_MEMORY = True
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING = False
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA = CfgNode()
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.START_EPS = 0.5
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.END_EPS = 0.05
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.T = 100
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.RESTART = False
_C.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.NON_EPS_TRAJS = 0
_C.GFN.TRAINING_DATA_LOADER.FIXED_SHAPE_TREES_PATH = \
    'phylo_likelihood_0725/ds1_fixed_shape_trees.p' 
# gfn model details
_C.GFN.MODEL = CfgNode()
_C.GFN.MODEL.ARCH = 'TRANSFORMER'
_C.GFN.MODEL.SEQ_LEN = 'fixed_seqlen'

_C.GFN.MODEL.Z_MLP = CfgNode()
_C.GFN.MODEL.Z_MLP.INPUT_SIZE = 128
_C.GFN.MODEL.Z_MLP.HIDDEN_SIZE = 256
_C.GFN.MODEL.Z_MLP.OUTPUT_SIZE = 1
_C.GFN.MODEL.Z_MLP.ACT_FN = 'RELU'
_C.GFN.MODEL.Z_MLP.LAYERS = 3
_C.GFN.MODEL.Z_MLP.DROPOUT = 0.0

# MLP for embedding the scale
_C.GFN.MODEL.SCALE_MLP = CfgNode()
_C.GFN.MODEL.SCALE_MLP.INPUT_SIZE = 1
_C.GFN.MODEL.SCALE_MLP.HIDDEN_SIZE = 256
_C.GFN.MODEL.SCALE_MLP.OUTPUT_SIZE = 128
_C.GFN.MODEL.SCALE_MLP.ACT_FN = 'RELU'
_C.GFN.MODEL.SCALE_MLP.LAYERS = 3
_C.GFN.MODEL.SCALE_MLP.DROPOUT = 0.0

# Transformer in GFN
_C.GFN.MODEL.TRANSFORMER = CfgNode()
_C.GFN.MODEL.TRANSFORMER.USE_TREE_TYPE_EMBEDDING = True
_C.GFN.MODEL.TRANSFORMER.SHARED_ENCODER = True
_C.GFN.MODEL.TRANSFORMER.NUM_HEADS = 4
_C.GFN.MODEL.TRANSFORMER.DEPTH = 6  # number of sa blocks
_C.GFN.MODEL.TRANSFORMER.MLP_RATIO = 2  # parmas in the mlp module in transformer block
_C.GFN.MODEL.TRANSFORMER.DROP_RATE = 0.0  # dropout rate
_C.GFN.MODEL.TRANSFORMER.ATTN_DROP_RATE = 0.0  # attention block dropout rate

# part 1 model options
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD = CfgNode()
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN = True
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.INPUT_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.OUTPUT_SIZE = 1
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.LAYERS = 3
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.DROPOUT = 0.0
_C.GFN.MODEL.TRANSFORMER.PART1_HEAD.ACT_FN = 'RELU'

# part 2 model options
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD = CfgNode()
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.CONCATENATE_CANDIDATE_TREE = True
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.INPUT_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.OUTPUT_SIZE = 1
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.LAYERS = 3
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.DROPOUT = 0.0
_C.GFN.MODEL.TRANSFORMER.PART2_HEAD.ACT_FN = 'RELU'

# forward policy logits for the single step model
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD = CfgNode()
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.CONCATENATE_SUMMARY_TOKEN = True
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.INPUT_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.OUTPUT_SIZE = 1
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.LAYERS = 3
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.DROPOUT = 0.0
_C.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.ACT_FN = 'RELU'

# SUBTB flow head
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD = CfgNode()
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.INPUT_SIZE = 128
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.OUTPUT_SIZE = 1
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.LAYERS = 3
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.DROPOUT = 0.0
_C.GFN.MODEL.TRANSFORMER.FLOW_HEAD.ACT_FN = 'RELU'

# sequence embedding at the start of the transformer input
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB = CfgNode()
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.INPUT_SIZE = 5 * 118
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.OUTPUT_SIZE = 128
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.HIDDEN_SIZE = 256
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.LAYERS = 0
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.DROPOUT = 0.0
_C.GFN.MODEL.TRANSFORMER.SEQ_EMB.ACT_FN = 'RELU'

# edge modeling
_C.GFN.MODEL.EDGES_MODELING = CfgNode()
_C.GFN.MODEL.EDGES_MODELING.DISTRIBUTION = 'CATEGORICAL'

_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL = CfgNode()
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.BIN_SIZE = 0.002
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.BINS = 20
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD = CfgNode()
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.INPUT_SIZE = 384
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.LAYERS = 3
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.OUTPUT_SIZE = 400
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.DROPOUT = 0.0
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.HEAD.ACT_FN = 'RELU'

_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD = CfgNode()
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.INPUT_SIZE = 384
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.HIDDEN_SIZE = 256
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.LAYERS = 3
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.OUTPUT_SIZE = 20
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.DROPOUT = 0.0
_C.GFN.MODEL.EDGES_MODELING.CATEGORICAL.ROOT_EDGE_HEAD.ACT_FN = 'RELU'

# training params
_C.GFN.MODEL.SUBTB_LAMBDA = 0.9
_C.GFN.MODEL.USE_TARGET_NET = False
_C.GFN.MODEL.TARGET_NET_UPDATE_FREQ = 200
_C.GFN.MODEL.Z_PARTITION_INIT = 5
_C.GFN.MODEL.LR_MODEL = 5e-5
_C.GFN.MODEL.LR_Z = 5e-3
_C.GFN.MODEL.GRAD_CLIP = 10
_C.GFN.MODEL.L2_REG = 0
_C.GFN.MODEL.LOSS_FN = 'MSE'

# evaluation params
_C.GFN.MODEL.EVALUATION = CfgNode()
_C.GFN.MODEL.EVALUATION.STATES_NUM = 100
_C.GFN.MODEL.EVALUATION.STATES_GENERATION_METHOD = 'UNIFORM_BINS'
_C.GFN.MODEL.EVALUATION.SAME_TREE_STRUCTURE = False
_C.GFN.MODEL.EVALUATION.BINS_NUM = 5
_C.GFN.MODEL.EVALUATION.MAX_DUPLICATE_MUTATIONS = 5
_C.GFN.MODEL.EVALUATION.TRAJECTORIES_PER_STATES = 1000
_C.GFN.MODEL.EVALUATION.EVALUATION_FREQ = 1
_C.GFN.MODEL.EVALUATION.FIXED_STATES = True
_C.GFN.MODEL.EVALUATION.PROB_ESTIMATION_METHOD = 'IMPORTANCE_SAMPLING'
_C.GFN.MODEL.EVALUATION.MUTATIONS_TRAJS = 10000
_C.GFN.MODEL.EVALUATION.NUM_WORKERS = 8
_C.GFN.MODEL.EVALUATION.BATCH_SIZE = 64
_C.GFN.MODEL.EVALUATION.PIN_MEMORY = True

# learning rate scheduler
_C.GFN.MODEL.USE_LR_SCHEDULER = False
_C.GFN.MODEL.LR_SCHEDULER = CfgNode()
_C.GFN.MODEL.LR_SCHEDULER.TYPE = 'COSINE_WITH_RESTART'

_C.GFN.MODEL.LR_SCHEDULER.COSINE_WITH_RESTART = CfgNode()
_C.GFN.MODEL.LR_SCHEDULER.COSINE_WITH_RESTART.LR_MIN = [5e-5, 5e-3]
_C.GFN.MODEL.LR_SCHEDULER.COSINE_WITH_RESTART.LR_MAX = [5e-4, 5e-2]
_C.GFN.MODEL.LR_SCHEDULER.COSINE_WITH_RESTART.T0 = 10
_C.GFN.MODEL.LR_SCHEDULER.COSINE_WITH_RESTART.CYCLE_MULTI = 1.0

_C.GFN.MODEL.LR_SCHEDULER.COSINE = CfgNode()
_C.GFN.MODEL.LR_SCHEDULER.COSINE.LR_MIN = [5e-5, 5e-3]
_C.GFN.MODEL.LR_SCHEDULER.COSINE.LR_MAX = [5e-4, 5e-2]
_C.GFN.MODEL.LR_SCHEDULER.COSINE.T_MAX = 10

_C.GFN.MODEL.LR_SCHEDULER.LINEAR = CfgNode()
_C.GFN.MODEL.LR_SCHEDULER.LINEAR.START_FACTOR = 1.0
_C.GFN.MODEL.LR_SCHEDULER.LINEAR.END_FACTOR = 0.1
_C.GFN.MODEL.LR_SCHEDULER.LINEAR.T = 30

# ENV
_C.ENV = CfgNode()
_C.ENV.SEQUENCE_TYPE = 'RNA_WITH_GAP'  # DNA DNA_WITH_GAP RNA RNA_WITH_GAP
_C.ENV.ENVIRONMENT_TYPE = 'TWO_STEPS_BINARY_TREE'

# ENN reward
_C.ENV.REWARD = CfgNode()
_C.ENV.REWARD.RESHAPE_METHOD = 'C-MUTATIONS'
_C.ENV.REWARD.C = 299.0
_C.ENV.REWARD.POWER = 1.0
_C.ENV.REWARD.SCALE = 1.0
_C.ENV.REWARD.EXP_MIN = 5e-324  # smallest number above 0 that is representable by float64
_C.ENV.REWARD.EXP_MAX = 8e307  # largest number that is representable by float64

# ENN evolution model for the likelihood problem
_C.ENV.EVOLUTION_MODEL = CfgNode()
_C.ENV.EVOLUTION_MODEL.COMPUTE_PRIOR = True
_C.ENV.EVOLUTION_MODEL.PRIOR_LAMBDA = 10.0
_C.ENV.EVOLUTION_MODEL.SEQUENCE_LENGTH = 1949
_C.ENV.EVOLUTION_MODEL.VOCAB_SIZE = 4

# logging
_C.LOGGING = CfgNode()
_C.LOGGING.ENABLE_TENSORBOARD = True
_C.LOGGING.TB_DIR = ''
_C.LOGGING.TB_NAME = 'tb_logs'

# Full experiment
_C.OUTPUT_PATH = ''


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for my_project.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern recommended by the YACS repo.
    # It will be subsequently overwritten with local YAML.
    return _C.clone()
