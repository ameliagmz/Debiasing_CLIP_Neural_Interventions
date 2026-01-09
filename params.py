import os

def setup_experiment(exp_name, EXP_BASE_DIR, IMGS_FOLDER, ATTN_FOLDER, AVG_ATTN_FOLDER, SISPI_PATH, BACKBONE, CLIP_PRETRAINED, AP_THRESHOLD, MAX_IMAGES):
    
    if not os.path.exists(EXP_BASE_DIR):
        os.makedirs(EXP_BASE_DIR)

    # --- Create main experiment folder ---
    EXP_DIR = os.path.join(EXP_BASE_DIR, exp_name)
    os.makedirs(EXP_DIR, exist_ok=True)

    # --- Create other folders ---
    attn_folder_path = os.path.join(EXP_DIR, ATTN_FOLDER)
    os.makedirs(attn_folder_path, exist_ok=True)
    avg_attn_path = os.path.join(EXP_DIR, AVG_ATTN_FOLDER)
    os.makedirs(avg_attn_path, exist_ok=True)

    # --- Save parameters ---
    params = {
        "EXP_NAME": exp_name,
        "IMGS_FOLDER": IMGS_FOLDER,
        "ATTN_FOLDER": ATTN_FOLDER,
        "AVG_ATTN_FOLDER": AVG_ATTN_FOLDER,
        "SISPI_PATH": SISPI_PATH,
        "BACKBONE": BACKBONE,
        "CLIP_PRETRAINED": CLIP_PRETRAINED,
        "AP_THRESHOLD": AP_THRESHOLD,
        "MAX_IMAGES": MAX_IMAGES
    }

    params_txt = os.path.join(EXP_DIR, "params.txt")
    with open(params_txt, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print(f"Experiment folder created: {EXP_DIR}")
    print(f"Parameters saved to: {params_txt}")

    return EXP_DIR, attn_folder_path, avg_attn_path


BASE_DIR = 'MODULAR_CODE'
EXP_BASE_DIR = os.path.join(BASE_DIR, 'experiments')

IMGS_FOLDER = 'FAIRFACE125train' # COCO2014_gender, 'COCOtrain2014_gender'
ATTN_FOLDER = 'attn_FF125'
AVG_ATTN_FOLDER = 'avg_attn_FF125'
SISPI_PATH = 'SISPIfork/data/images'
os.makedirs(ATTN_FOLDER, exist_ok=True)

BACKBONE = 'ViT-B-16' # 'ViT-B-32'
CLIP_PRETRAINED = 'openai' # laion2b_s34b_b79k, openai
AP_THRESHOLD = 0.75
MAX_IMAGES = 10000
MODEL_TYPE = 'intervention'
TEST_ATTRIBUTE = "gender" 
FF_CROP = "1.25"

exp_name = f'{BACKBONE}_{CLIP_PRETRAINED}_{AP_THRESHOLD}_{MAX_IMAGES}'

EXP_DIR, ATTN_DIR, AVG_ATTN_DIR = setup_experiment(exp_name, EXP_BASE_DIR, IMGS_FOLDER, ATTN_FOLDER, AVG_ATTN_FOLDER, SISPI_PATH, BACKBONE, CLIP_PRETRAINED, AP_THRESHOLD, MAX_IMAGES)