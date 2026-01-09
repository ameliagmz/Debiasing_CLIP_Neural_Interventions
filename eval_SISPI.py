import os
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
from types import MethodType
import open_clip
from PIL import Image
from sklearn.metrics import ndcg_score, average_precision_score
from open_clip import tokenizer
import reranking
from tqdm import tqdm
import re
import json
from intervention import replace_forward_only
from params import BACKBONE, CLIP_PRETRAINED

def get_profession_names(SISPI_path):
    words = set()

    # Iterate through all the files in the folder
    for filename in os.listdir(SISPI_path):
        # Check if the file is an image (optional, to handle only image files)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Split the filename at the first underscore and take the first part
            first_word = filename.split('_')[0]
            words.add(first_word)  # Add the first word to the set

    return list(words)

def get_profession_images(profession, SISPI_path):
    labels = []
    images = []

    for filename in os.listdir(SISPI_path):
        if profession in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path of the image
            full_path = os.path.join(SISPI_path, filename)
            
            # Open the image, convert to RGB, and append to the list
            img = Image.open(full_path).convert("RGB")
            if 'female' in filename:
                labels.append(1)
            elif 'male' in filename:
                labels.append(0)

            images.append(img)

    # Now 'images' contains all the opened images in RGB format

    article = "an" if profession[0].lower() in "aeiou" else "a"
    prompt = f"a photo of {article} {profession}"

    return images, labels, prompt


def compute_ndkl(image_features, text_features, labels):

    image_features = torch.stack(image_features, dim=0).squeeze(1)
    similarity_scores = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity_scores = similarity_scores.flatten()

    sorted_indices = np.argsort(-similarity_scores)
    sorted_indices = sorted_indices.flatten()

    # Sort the labels based on these sorted indices
    sorted_labels = [labels[i] for i in sorted_indices]
    desired_distribution = {0: 0.5, 1: 0.5}
    ndkl = reranking.ndkl(sorted_labels, desired_distribution)

    return ndkl

def compute_map(image_features, text_features):
    map_int_dic = {}
    ndcg_int_dic = {}

    # Prepare a global gallery of all image features
    all_image_features = []
    all_labels = []
    for profession, feats in image_features.items():
        all_image_features.extend(feats)
        all_labels.extend([profession] * len(feats))

    all_image_features = torch.stack(all_image_features, dim=0).squeeze(1).cpu().numpy()
    all_labels = np.array(all_labels)  # Shape: (num_images,)

    for profession in image_features.keys():
        query_feature = text_features[profession]
        # Compute similarities with all images in the gallery
        similarities = (query_feature.cpu().numpy() @ all_image_features.T).squeeze()

        # Create binary relevance labels: 1 for relevant (same profession), 0 otherwise
        relevance_labels = (all_labels == profession).astype(int)

        # Compute Average Precision (AP)
        map_int_dic[profession] = average_precision_score(relevance_labels, similarities)
        ndcg_int_dic[profession] = ndcg_score([relevance_labels], [similarities])

    return map_int_dic, ndcg_int_dic


def exp_eval(layer, SISPI_path, heads, new_attentions):
    ndkl_int_dic = {}
    map_int_dic = {}
    professions = get_profession_names(SISPI_path)

    model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE, pretrained=CLIP_PRETRAINED)
    model.eval()
    model.to('cuda')

    if layer > 0:
        # Replace the forward funtion of all MultiheadAttention layers in the model
        replace_forward_only(model,layer,heads,new_attentions)


    # Extract features for all images/texts
    image_features = {}
    text_features  = {}
    labels = {}
    with torch.no_grad():

        for profession in tqdm(professions, desc='Extracting features', position=0):
            images, labels[profession], text_prompt = get_profession_images(profession, SISPI_path)
            text_features[profession] = model.encode_text(tokenizer.tokenize(text_prompt).to('cuda'))
            text_features[profession] /= text_features[profession].norm(dim=-1, keepdim=True)
    
            image_features[profession] = []
            for image in images:
                image = preprocess(image).unsqueeze(0)
                image_feature = model.encode_image(image.to('cuda'))
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                image_features[profession].append(image_feature)

            ndkl = compute_ndkl(image_features[profession], text_features[profession], labels[profession])
            ndkl_int_dic[profession] = ndkl

        map_int_dic, ndcg_int_dic = compute_map(image_features, text_features)

    return ndkl_int_dic, map_int_dic, ndcg_int_dic


def SISPI_eval(new_attn_exp, heads_exp, layers_exp, interventions_exp, SISPI_PATH, EXP_DIR):
    ndkl_results = {}
    map_results = {}
    ndcg_results = {}
    summary_results = {}

    for new_attentions, heads, layer, interv in zip(new_attn_exp, heads_exp, layers_exp, interventions_exp):

        exp_key = '_'.join(['layer',str(layer),'heads']+[str(h) for h in heads]+[interv])
        subtitle = f"Layer: {layer}, Heads: {[i for i in heads]}, Intervention: {interv}"
        print('/'*40,'\n',subtitle)

        ndkl_results[exp_key], map_results[exp_key], ndcg_results[exp_key]  = exp_eval(layer, SISPI_PATH, heads, new_attentions)

        mean_ndkl = np.mean(list(ndkl_results[exp_key].values()))
        mean_map = np.mean(list(map_results[exp_key].values()))
        mean_ndcg = np.mean(list(ndcg_results[exp_key].values()))

        # Print metrics
        print(f'NDCG: {mean_ndcg:.4f}')
        print(f'mAP: {mean_map:.4f}')
        print(f'NDKL: {mean_ndkl:.4f}')

        summary_results[exp_key] = {
        "mean_NDKL": mean_ndkl,
        "mean_mAP": mean_map,
        "mean_NDCG": mean_ndcg}

    # Save results of the experiments
    with open(os.path.join(EXP_DIR, 'ndkl_SISPI_maleWhite.json'), 'w') as file:
        json.dump(ndkl_results, file)

    with open(os.path.join(EXP_DIR, 'map_SISPI_maleWhite.json'), 'w') as file:
        json.dump(map_results, file)

    with open(os.path.join(EXP_DIR, 'ndcg_SISPI_maleWhite.json'), 'w') as file:
        json.dump(ndcg_results, file)

    with open(os.path.join(EXP_DIR, 'summary_SISPI_maleWhite.json'), 'w') as file:
        json.dump(summary_results, file, indent=2)