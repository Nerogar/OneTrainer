#!/bin/env python

"""
A util to print out what image directories a subdir under 
your "workdir-cache" may use.
This doesnt catch ALL usage -- currently, it only looks at
the "variation-0" subdirectory.
Although you can manually call it on each "variation" directory
if you wish, but it should be the same?

This is useful if you have multiple datadirs cached, and want to remove
ONE of them, instead of your entire data cache

Example:  
    python cache-info.py /OneTrainer/workdir-cache/abcdefreallylongnamehere

"""


import torch
import os,sys

def summarize_image_directories(aggregate_file):
    data = torch.load(aggregate_file,weights_only=True)
    
    if not isinstance(data, list):
        raise TypeError("The aggregate.pt file does not contain a list of dictionaries.")
    
    image_paths = []
    for entry in data:
        if 'image_path' in entry:
            image_paths.append(entry['image_path'])
        else:
            raise KeyError("One of the entries does not contain 'image_path'.")
    
    directories = {os.path.dirname(path) for path in image_paths}
    
    sorted_directories = sorted(directories)
    
    print("Unique directories found:")
    for directory in sorted_directories:
        print(directory)
    
    return sorted_directories

if len(sys.argv) <2:
    aggregate_file = "aggregate.pt"
else:
    directory = sys.argv[1]
    if os.path.exists(os.path.join(directory, "variation-0")):
        directory = os.path.join(directory, "variation-0")
    aggregate_file = os.path.join(directory, "aggregate.pt")

unique_directories = summarize_image_directories(aggregate_file)

