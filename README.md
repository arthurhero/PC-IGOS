# PC-IGOS
Point Cloud Integrated-Gradients Optimized Saliency, a visualization technique that can automatically find the minimal saliency map that covers the most important features on a shape.

# Author(s)

Ziwen Chen, Wenxuan Wu, Zhongang Qi

## Usage

### Preparing ModelNet40
1. Download and unzip the [modelnet40_normal_resampled](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) dataset to a `data` folder.
2. Create an empty checkpoint log folder for the classifier `mkdir log`.

### Training the PointConv classifier
1. For training a PointConv point cloud classifier, run `python3 train_pointconv.py`.
2. If the training flattened, stop the process and tune down the `lr` hyperparam.

### Precompute blurred shapes for ModelNet40 test split
1. Run `python3 modelnetdataset.py`. It will run the `save_all_blurs()` method defined there.
2. If you want to save point clouds along the del/ins curves, create a folder `mkdir tensors` and toggle `visualize` to be true in `evaluate_on_all_classes()`.

### Evaluating PC-IGOS on ModelNet40 test split
1. Run `python3 pc_IGOS.py`. The main code for PC-IGOS is `integrate_mask()`.


## Files

*blur_utils.py*
:   utils for smoothing curvatures on point clouds

*modelnetdataset.py*
:   PyTorch datasets for ModelNet40

*pc_IGOS.py*
:   main PC-IGOS code and evaluation code

*pointconv.py*
:   architecture of a PointConv classifier

*pointconv_utils.py*
:   utils for point cloud classifers

*train_pointconv.py*
:   training code for the PointConv classifier
