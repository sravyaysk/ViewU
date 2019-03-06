# ViewU
ViewU is an automated tool, that can enable user to have of view of him/her on the products he/she purchase like dresses, accessories, jewelry etc.This is implemented using GANs

# Dependency
- ```pytorch >= 0.4.0```
- [visdom](https://github.com/facebookresearch/visdom).
- [opencv](https://github.com/opencv/opencv)

# Training

To train the model, we used LOOKBOOK dataset, resize images to 64*64. Prepare the dataset using `pytorch-GAN/tool/prepare_data.py`.
Then goto src dir and run
```
python3 train.py
```

# Monitor the performance


- Install [visdom](https://github.com/facebookresearch/visdom).
- Start the visdom server with ```python3 -m visdom.server 5274```
- Open this URL in your browser: `http://localhost:5274` You will see the loss curve as well as the image examples.


- Development in Progress
