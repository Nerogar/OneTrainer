# Things to add

OneTrainer is far from done. Here is a list of things that I would love to implement, but didn't find the time yet.

- EMA model: This would make it possible to train on many different concepts with huge data sets.
- Full safetensors support: Currently, only a few operations support the safetensors format. It would be great if all
  model loaders and model savers would support the format.
- Support for more base models: Only the base Stable Diffusion 1.5 and 1.5 Inpainting models are supported at the
  moment. The goal would be to support all currently released model types.
- Accelerate support for multi GPU training: This will be a bit more complicated.
  [MGDS](https://github.com/Nerogar/MGDS), the data loader library, needs to be thread safe first.

Some ideas that need to be implemented in [MGDS](https://github.com/Nerogar/MGDS):

- Tag shuffling
- Tag/Caption dropout
- Support for loading concepts from nested folder structures
- Better caching: Make it possible somehow to add and remove concepts or samples from the cache without rebuilding the
  whole cache. Maybe by splitting the cache into multiple folders based on concepts.
- Adding to that, make it possible to cache multiple versions of a concept within one epoch. This could be used to
  increase variations of an underrepresented concept.
- Support for class images and prior loss preservation for each concept
