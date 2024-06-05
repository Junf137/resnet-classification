# global variables
DENS_LEVEL = 10
PARTITION_VERSION = 2  # 1 or 2

# paths
BASE_FOLDER = "../_DatasetsLocal/CompoundEyeClassification/Data/"
MODEL_PATH = "../_ModelsLocal/resnet-50"
SAVE_WEIGHTS = "./models/best_model_ver_" + str(PARTITION_VERSION) + ".pth"
IMG_PATH = "./images/ver_" + str(PARTITION_VERSION) + "/"
VALID_EXTENSIONS = {".png", ".jpg"}

# training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_CLASSES = 3
TRAINING_PORTION = 0.8
VALIDATION_PORTION = 0.1
EPOCHS = 2

# data partition
LABELS_FOLDER_NUM = {"E": 8, "I": 4, "O": 4}
LABELS_DIRS_INDEX = {"E": [], "I": [], "O": []}

# training control
FAST_DEBUG = False
CONTINUE_TRAINING = True
LOAD_INDICES = True
