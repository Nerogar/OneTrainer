# Embedding Training

To get a general overview of the UI, please read the [Quick Start Guide](QuickStartGuide.md) first.

Training an embedding requires a few different settings than fine-tuning or LoRA training. You can enable Embedding
training in the dropdown in the top right corner.

### Concepts

To train an embedding, you need to use special prompts. Each prompt needs to include `<embedding>` in the place where
you want to place your trained embedding. For example, if you want to train a style of a painting, your prompt could
be `a painting in the style of <embedding>`. If you don't want to add a custom prompt for every training image, you can
select "From single text file" as the prompt source of your concept. Then select a text file containing one prompt per
line. An example of such a file can be found in the embedding_templates directory.

### Special Embedding Settings

If you select "Embedding" as your training method, a new tab called "embedding" will appear. Here you can specify:

- Base embedding: An already trained embedding you want to continue training on. Leave this blank to train a new
  embedding
- Token count: The number of tokens to train. A higher token count is better at learning things, but will also take up
  more of the available tokens in each prompt when generating an image.
- Initial embedding text: The text to use when initializing a new embedding. Choosing a good text can significantly
  speed up training.
