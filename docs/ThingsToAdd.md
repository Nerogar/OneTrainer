# Things to add

OneTrainer is far from done. Here is a list of things that I would love to implement, but didn't find the time yet.

- Learning rate schedulers: Only a constant learning rate is supported right now. For more complex training tasks,
  different schedulers, including learning rate warmup is needed. This also includes saving the current state when
  taking a backup
- EMA model: This would make it possible to train on many different concepts with huge data sets.
- Full safetensors support: Currently, only a few operations support the safetensors format. It would be great if all
  model loaders and model savers would support the format.
- Support for more base models: Only the base Stable Diffusion 1.5 and 1.5 Inpainting models are supported at the
  moment. The goal would be to support all currently released model types.
- Accelerate support for multi GPU training: This will be a bit more complicated.
  [MGDS](https://github.com/Nerogar/MGDS), the data loader library, needs to be thread safe first.
- Full Pytorch2 support: Pytorch2 enables the use of SDP-Attention, which should be able to replace XFormers. It should
  be straight forward to upgrade, but in my limited tests, There were always some limitations like high memory usage, or
  crashes.

Some ideas that need to be implemented in [MGDS](https://github.com/Nerogar/MGDS):

- Tag shuffling
- Tag/Caption dropout
- Support for loading concepts from nested folder structures
- Better caching: Make it possible somehow to add and remove concepts or samples from the cache without rebuilding the
  whole cache. Maybe by splitting the cache into multiple folders based on concepts.
- Support for multiple prompts per image sample
- Support for class images and prior loss preservation for each concept