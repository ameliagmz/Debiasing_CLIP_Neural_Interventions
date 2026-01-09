import clip_debiasing
import open_clip
import torch
import clip
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import json
import os

from clip_debiasing.models.model_clipped import CLIP_clipped
from clip_debiasing.models.model_prompt_gender import CLIP_prompt_gender
from clip_debiasing.models.model_prompt_age import CLIP_prompt_age
from clip_debiasing.models.model_prompt_race import CLIP_prompt_race


from clip_debiasing.models.model_vl_debiasing import DebiasedCLIP

from intervention import replace_forward_only

from eval_retrieval import create_dataset, create_loader, evaluation, itm_eval

import numpy as np

# Suppress divide by zero and invalid value warnings
np.seterr(divide='ignore', invalid='ignore')


def run_eval(model, preprocess, attribute="gender", device=torch.device('cuda'), no_batches=False):
    model.eval()
    if no_batches:
        bias_result_fairface = clip_debiasing.measure_bias_no_batches(model, preprocess, clip.tokenize, attribute=attribute, dataset='fairface')

    else:
        bias_result_fairface = clip_debiasing.measure_bias(model, preprocess, clip.tokenize, attribute=attribute, dataset='fairface')
    # print(bias_result_fairface)

    return bias_result_fairface

def eval_zhang(device, model_type, test_attribute, backbone, EXP_DIR, interv=None):
    if model_type == 'original':
        model, preprocess = clip.load(backbone, device=device)
        run_eval(model, preprocess,
             attribute=test_attribute
             )

    elif model_type == 'CLIP-clip':
        hidden_dim = 768 if backbone == 'ViT-L/14' else 1024 if backbone == 'RN50' else 512
        if backbone == 'ViT-H/14':
            dims = [512, 800, 980]
        elif backbone == 'ViT-L/14':
            dims = [384, 600, 735]
        else:
            dims = [256, 400, 490]
        for dim in dims:
            model = CLIP_clipped(backbone, device=device, hidden_dim=hidden_dim, m=dim, attribute=test_attribute)
            preprocess = model.preprocess
            run_eval(model, preprocess,
                attribute=test_attribute
                )

    elif model_type == 'biased-prompt':
        model = CLIP_prompt_gender(backbone, device=device) if test_attribute=="gender" else CLIP_prompt_age(backbone, device=device) if test_attribute=='age' else CLIP_prompt_race(backbone, device=device)
        preprocess = model.preprocess
        run_eval(model, preprocess,
             attribute=test_attribute
             )

    elif model_type == 'vl_debiasing':
        model = DebiasedCLIP(backbone, device=device, mlp1_hidden_size=512, mlp2_hidden_size=1024) # change the debiasing configs
        preprocess = model.preprocess
        model.load_state_dict(torch.load("vit_b_16_fairface_gender_renamed.pth")) # change the path
        run_eval(model, preprocess,
             attribute=test_attribute
             )
        
    elif model_type == 'intervention':

        (new_attn_exp, heads_exp, layers_exp, interventions_exp) = interv
        
        results_dict = {}

        for new_attentions, heads, layer, interv in zip(new_attn_exp, heads_exp, layers_exp, interventions_exp):
            exp_key = '_'.join(['layer',str(layer),'heads']+[str(h) for h in heads]+[interv])
            subtitle = f"Layer: {layer}, Heads: {[i for i in heads]}, Intervention: {interv}"
            print('/'*40,'\n',subtitle)

            model, preprocess = clip.load(backbone, device=device)

            model.eval()
            model.to('cuda')

            if layer > 0:
                # Replace the forward funtion of all MultiheadAttention layers in the model
                replace_forward_only(model,layer,heads,new_attentions)

            results = run_eval(model, preprocess, attribute=test_attribute, no_batches=True)

            results_dict[exp_key] = results

            print('/'*40,'\n',subtitle)
            print(results)
            print('\n\n')


        with open(os.path.join(EXP_DIR, "zhang_eval_maleWhite.json"), 'w') as file:
            json.dump(results_dict, file)