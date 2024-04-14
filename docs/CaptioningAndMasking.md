# Captioning and Masking

OneTrainer includes a UI for captioning and masking of your dataset. To access it, open the "Dataset Tools" on the tools
tab. Once the UI is open, click the "Open" button on the top left, and select the directory of your dataset.

### Navigating the UI

To switch between the images, either click on the filename in the list on the left side, or use the up and down arrow
keys to go to the next or previous image.

### Manual captioning

In the input box at the bottom you can input your caption. To save the caption, press enter.

### Automatic captioning

Click on the "Generate Captions" button to open the batch captioning tool. Here you can choose which model to use for
captioning. For the initial caption, you can choose a text that should be used to start the new caption. To generate the
captions, press the "Create Captions" button. When you use this for the first time, it has to download the model.
Depending on the model you chose, this can take a while.

### Manual masking

Check the "Enable Mask Editing" checkbox at the top. Now you can draw a mask onto the image. Left click adds to the
masked region, right click removes parts from the mask. With the mouse wheel you can increase or decrease the brush
size. Use Ctrl+M to only show the mask. To save the mask, click into the caption input field, then press enter.

### Automatic masking

Click on the "Generate Masks" button to open the batch captioning tool. Here you can choose which model to use for
masking. Some models like ClipSeg support masking based on a Prompt. With Threshold, Smooth and Expand you need to
experiment around to find what works best for your dataset. To generate the masks, press the "Create Masks" button. When
you use this for the first time, it has to download the model. Depending on the model you chose, this can take a while.