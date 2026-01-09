import os
import numpy as np
import torch
import re

# Get average attention given an attribute (male, female)
def get_attentions_attribute(attentions_folder,target_attribute,layer,head):
    layers = ['layer9_attn', 'layer10_attn', 'layer11_attn', 'layer12_attn']
    
    all_activations = []

    attention_files = [f for f in os.listdir(attentions_folder) if f.endswith('.npy')]

    for file in attention_files:
        if f"_{target_attribute}" in file: # Attributes always follow a _, avoid obtaining "female" files when searching for "male"
            file_path = os.path.join(attentions_folder, file)
            activations = np.load(file_path)
            attn = activations[layers.index(layer), head-1, :]  # Extract attention for specified layer and head
            all_activations.append(attn)

    all_activations = np.array(all_activations)
    average_tensor = np.mean(all_activations, axis=0)  # Average of each element across all tensors

    average_tensor = torch.tensor(average_tensor, dtype=torch.float32)

    return average_tensor


# Get average attention of person
def get_attentions_all(attentions_folder,layer,head):
    layers = ['layer9_attn', 'layer10_attn', 'layer11_attn', 'layer12_attn']
    
    all_activations = []

    attention_files = [f for f in os.listdir(attentions_folder) if f.endswith('.npy')]

    for file in attention_files:
        file_path = os.path.join(attentions_folder, file)
        activations = np.load(file_path)
        attn = activations[layers.index(layer), head-1, :]  # Extract attention for specified layer and head
        all_activations.append(attn)

    all_activations = np.array(all_activations)
    average_tensor = np.mean(all_activations, axis=0)  # Average of each element across all tensors

    average_tensor = torch.tensor(average_tensor, dtype=torch.float32)

    return average_tensor


def get_avg_attns_gender(input_attn_folder, avg_attn_folder):
    for l in ['layer9_attn','layer10_attn','layer11_attn','layer12_attn']:
        for h in range(1,13):
            layer_num = int(re.search(r'\d+', l).group())
            name = f"_l{layer_num}_h{h}"
            person = get_attentions_all(input_attn_folder,l,h)

            male = get_attentions_attribute(input_attn_folder,"male",l,h) 
            female = get_attentions_attribute(input_attn_folder,"female",l,h)
            
            torch.save(person, os.path.join(avg_attn_folder, 'person'+name+'.pt'))
            torch.save(male, os.path.join(avg_attn_folder, 'male'+name+'.pt'))
            torch.save(female, os.path.join(avg_attn_folder, 'female'+name+'.pt'))


def get_avg_attns_ethnicity_ff(input_attn_folder, avg_attn_folder):
    for l in ['layer9_attn','layer10_attn','layer11_attn','layer12_attn']:
        for h in range(1,13):
            layer_num = int(re.search(r'\d+', l).group())
            name = f"_l{layer_num}_h{h}"
            # person = get_attentions_all(input_attn_folder,l,h)

            # east_asian = get_attentions_attribute(input_attn_folder,"EastAsian",l,h) 
            # indian = get_attentions_attribute(input_attn_folder,"Indian",l,h)
            # black = get_attentions_attribute(input_attn_folder,"Black",l,h)
            # white = get_attentions_attribute(input_attn_folder,"White",l,h)
            # middle_eastern = get_attentions_attribute(input_attn_folder,"MiddleEastern",l,h)
            # latino = get_attentions_attribute(input_attn_folder,"LatinoHispanic",l,h)
            # southeast_asian = get_attentions_attribute(input_attn_folder,"SoutheastAsian",l,h)

            # torch.save(person, os.path.join(avg_attn_folder, 'person'+name+'.pt'))
            # torch.save(east_asian, os.path.join(avg_attn_folder, 'EastAsian'+name+'.pt'))
            # torch.save(indian, os.path.join(avg_attn_folder, 'Indian'+name+'.pt'))
            # torch.save(black, os.path.join(avg_attn_folder, 'Black'+name+'.pt'))
            # torch.save(white, os.path.join(avg_attn_folder, 'White'+name+'.pt'))
            # torch.save(middle_eastern, os.path.join(avg_attn_folder, 'MiddleEastern'+name+'.pt'))
            # torch.save(latino, os.path.join(avg_attn_folder, 'LatinoHispanic'+name+'.pt'))
            # torch.save(southeast_asian, os.path.join(avg_attn_folder, 'SoutheastAsian'+name+'.pt'))


            white = get_attentions_attribute(input_attn_folder,"White",l,h)
            torch.save(white, os.path.join(avg_attn_folder, 'White'+name+'.pt'))

