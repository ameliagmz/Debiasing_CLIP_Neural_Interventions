import torch
from attention_heads_class import SimpleMLP, MLP, get_best_heads
import pandas as pd
import re
from get_attentions import get_attn
from avg_attentions import get_avg_attns_gender, get_avg_attns_ethnicity_ff
import open_clip
from params import EXP_DIR, IMGS_FOLDER, ATTN_DIR, BACKBONE, CLIP_PRETRAINED, AP_THRESHOLD, AVG_ATTN_DIR, SISPI_PATH, MAX_IMAGES, MODEL_TYPE, TEST_ATTRIBUTE, FF_CROP
from intervention import generate_attention_experiments
from eval_SISPI import SISPI_eval
from zhang_eval import eval_zhang
from eval_SoBIT import eval_So_B_IT
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD CLIP
model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE, pretrained=CLIP_PRETRAINED)
model.eval()
model = model.to(device)

start_time = time.time()

# GENERATE ATTENTIONS
get_attn(model, preprocess, IMGS_FOLDER, ATTN_DIR, MAX_IMAGES) # get attentions of last 4 layers

# GENERATE AVG_ATTENTIONS
get_avg_attns_gender(ATTN_DIR, AVG_ATTN_DIR)
get_avg_attns_ethnicity_ff(ATTN_DIR, AVG_ATTN_DIR)

# HEADS TO EVALUATE
layers = ['layer9_attn', 'layer10_attn', 'layer11_attn', 'layer12_attn']
heads = [1,2,3,4,5,6,7,8,9,10,11,12]

head_classifier = SimpleMLP(768).to(device)

best_heads_dict, all_dfs = get_best_heads(head_classifier, layers, heads, ATTN_DIR, AP_THRESHOLD)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.5f} seconds")


for l, best_heads in best_heads_dict.items():

    # INTERVENTION
    new_attn_exp, heads_exp, layers_exp, interventions_exp = generate_attention_experiments(l, best_heads, AVG_ATTN_DIR)

    # EVALUATION
    SISPI_eval(new_attn_exp, heads_exp, layers_exp, interventions_exp, SISPI_PATH, EXP_DIR)

    interv = (new_attn_exp, heads_exp, layers_exp, interventions_exp)
    backbone = re.sub(r'-(\d+)$', r'/\1', BACKBONE)

    eval_zhang(device=device, model_type=MODEL_TYPE, test_attribute=TEST_ATTRIBUTE, backbone=backbone, EXP_DIR=EXP_DIR, interv=interv)

    eval_So_B_IT(model, preprocess, MODEL_TYPE, backbone, TEST_ATTRIBUTE, interv, FF_CROP)