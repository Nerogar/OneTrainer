# Project Structure

OneTrainer is mostly built on top of the Hugging Face [diffusers](https://github.com/huggingface/diffusers) library.

## Modules

The structure is split up into modules that each have a single responsibility. Different modules of the same type can
usually be interchanged. Each module type has its own sub-folder in the `modules` folder.

These are the currently supported module types:

- dataLoader: A data loader is responsible for loading samples during training. OneTrainer is using
  [MGDS](https://github.com/Nerogar/MGDS) as a library to implement these data loaders, which is a custom graph-based
  data loader implementation.
- model: A model holds weights, optimizers and related data that are needed during training.
- modelLoader: A model loader loads the model to train on into the internal representation. Model loaders support
  different model formats such as checkpoints and safetensors.
- modelSampler: A model sampler can sample a model either during training for a preview, or called separately to test
  your trained model.
- modelSaver: A model saver does the reverse of a model loader. It saves the model from an internal representation into
  a usable file format.
- modelSetup: A model setup provides different functions to set up the training process, without doing any training
  itself. These functions include creating an optimizer, moving the model to an appropriate device, running predictions
  on the model.
- trainer: A trainer pulls everything together. It runs a model loader and model setup, then executes the training loop.
  During the training, many different steps can be taken such as regular sampling or backups.
- ui: The ui module contains all the ui code.
- util: A set of utility functions needed in different parts of the code base.

## Scripts

Of course these modules don't provide any user facing functionality. Think of them as a set of tools that can be used by
scripts to provide actual functionality for a user. Each script has exactly one purpose and can be run directly from the
command line. Inside the scripts, no extra functionality is implemented. They should only rely on the modules for
functionality.
