CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Image Processing Constants
DEFAULT_IMAGE_SIZE = 336
DEFAULT_IMAGE_CHANNELS = 3
MAX_MODEL_LENGTH_SHORT = 5000
MAX_MODEL_LENGTH_LONG = 15000

# Layer Planning Constants
LAYER_MAPPING = {
    0: "Background",
    1: "Underlay",
    2: "Logo/Image",
    3: "Text",
    4: "Embellishment"
}

NUM_LAYERS = 5
