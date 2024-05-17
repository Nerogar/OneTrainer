import os

from copy import deepcopy
from pathlib import Path
import random
from typing import Any
from unittest import result
from uuid import uuid1
import chevron
from modules.util.config.SampleConfig import SampleConfig

from modules.util.config.TrainConfig import TrainConfig

def replace_text_in_trainconfig(original: TrainConfig, test_only: bool)  -> TrainConfig:
    new_train_config = deepcopy(original)
    new_train_config.workspace_dir = replace_text(new_train_config.workspace_dir, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
    new_train_config.cache_dir = replace_text(new_train_config.cache_dir, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
    new_train_config.output_model_destination = replace_text(new_train_config.output_model_destination, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
    instanceguid = str(uuid1())
    
    if new_train_config.concepts is not None:
        for c in new_train_config.concepts:
            c.name = replace_text(c.name, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
            c.path = replace_text(c.path, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
    else:
        concepts = read_file(original.concept_file_name)
        concepts = replace_text(concepts, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
        tempfilename = "%s_concepts.json" % (instanceguid)
        save_to_directory(concepts, new_train_config.workspace_dir, tempfilename)
        tempconceptsfilepath = os.path.join(new_train_config.workspace_dir, tempfilename)
        new_train_config.concept_file_name = tempconceptsfilepath
        

    if new_train_config.samples is not None:
        for s in new_train_config.samples:
            s.prompt = replace_text(c.name, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
    else:
        samples = read_file(original.sample_definition_file_name)
        samples = replace_text(samples, new_train_config.automation_replacement_keyword, new_train_config.automation_replacement_text)
        samplestempfilename = "%s_samples.json" % (instanceguid)
        save_to_directory(samples, new_train_config.workspace_dir, samplestempfilename)
        tempsamplesfilepath = os.path.join(new_train_config.workspace_dir, samplestempfilename)
        new_train_config.sample_definition_file_name = tempsamplesfilepath
    
    return new_train_config

def replace_text_in_sampleconfig(original: SampleConfig, keyword: str, replacement_text:str) -> SampleConfig:
    new_sample_config = deepcopy(original)
    new_sample_config.prompt = replace_text(c.name, keyword, replacement_text)
    return new_sample_config
        

def replace_text(original: str, keyword: str, replacement_text: str):
    return chevron.render(original, {keyword: replacement_text})

def replace_text_from_dict(original: str, replacement_text_dict: dict[str, Any]):
    return chevron.render(original, replacement_text_dict)

def read_file(file_path) -> str:
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None

def save_to_directory(content, output_dir, output_filename):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Saved modified content to: {output_path}")
    except Exception as e:
        print(f"Error while saving to directory: {e}")
        

def get_sub_directory_paths(start_directory):
    sub_directories = []
    for item in os.listdir(start_directory):
        full_path = os.path.join(start_directory, item)
        if os.path.isdir(full_path):
            sub_directories.append(full_path)
            # Recursive call to search through all subdirectories.
            get_sub_directory_paths(full_path)
    return sub_directories

def split_directory_names(directory_list):
    split_values = {}
    for directory in directory_list:
        folder_name = str(os.path.basename(directory))
        split_values[folder_name] = folder_name.split("-")
    return split_values

def parse_directory_for_folders(path_to_search: str, original: TrainConfig)  -> TrainConfig:
    all_directories = get_sub_directory_paths(path_to_search)
    result_dict = split_directory_names(all_directories)
    
    return result_dict

def process_parsed_directories(directories: dict[str, Any], original: TrainConfig, test_only: bool) -> list[TrainConfig]:
    results = []
    for directory, variablies in directories.items():
        keywords = {}
        counter = 1
        for var in variablies:
            var_str = f"V{counter}"
            keywords[var_str] = var
            counter += 1
        new_train_config = deepcopy(original)
        new_train_config = replace_text_in_trainconfig_from_dict(original, keywords, test_only)
        results.append(new_train_config)
    return results
              
def replace_text_in_trainconfig_from_dict(original: TrainConfig, keywords: dict[str, Any], test_only: bool)  -> TrainConfig:
    new_train_config = deepcopy(original)
    new_train_config.workspace_dir = replace_text_from_dict(new_train_config.workspace_dir, keywords)
    new_train_config.cache_dir = replace_text_from_dict(new_train_config.cache_dir,  keywords)
    new_train_config.output_model_destination = replace_text_from_dict(new_train_config.output_model_destination, keywords)
    instanceguid = str(uuid1())
    
    if new_train_config.concepts is not None:
        for c in new_train_config.concepts:
            c.name = replace_text_from_dict(c.name, keywords)
            c.path = replace_text_from_dict(c.path, keywords)
    else:
        concepts = read_file(original.concept_file_name)
        concepts = replace_text_from_dict(concepts, keywords)
        tempfilename = "concepts.json"
        if not test_only:
            save_to_directory(concepts, new_train_config.workspace_dir, tempfilename)
        tempconceptsfilepath = os.path.join(new_train_config.workspace_dir, tempfilename)
        new_train_config.concept_file_name = tempconceptsfilepath
        

    if new_train_config.samples is not None:
        for s in new_train_config.samples:
            s.prompt = replace_text_from_dict(c.name, keywords)
    else:
        samples = read_file(original.sample_definition_file_name)
        samples = replace_text_from_dict(samples, keywords)
        samplestempfilename = "samples.json"
        if not test_only:
            save_to_directory(samples, new_train_config.workspace_dir, samplestempfilename)
        tempsamplesfilepath = os.path.join(new_train_config.workspace_dir, samplestempfilename)
        new_train_config.sample_definition_file_name = tempsamplesfilepath
    
    return new_train_config
    