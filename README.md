# videoclassification
Use Tensforflow frozen graph for video classification

# TensorFlow C++ and Python Video Classification Demo

This example shows how you can load a pre-trained TensorFlow network and use it
to recognize objects in images/videos in Python/C++.

##TODO: Make changes to C++ to support video classification

## Description

This demo uses a Google Inception model to classify image/video files that are passed
in on the command line.

## Clone this repo
```bash
$ git clone https://github.com/M-Melodious/videoclassification.git
$ cd videoclassification
```

## To build/install/run

The TensorFlow `GraphDef` that contains the model definition and weights is not
packaged in the repo because of its size. Instead, you must first download the
file to the `data` directory in the source tree:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C videoclassification/label_image/data -xz
```

Then, as long as you've managed to build the main TensorFlow framework, you
should have everything you need to run this example installed already.

Once extracted, see the labels file in the data directory for the possible
classifications, which are the 1,000 categories used in the Imagenet
competition.

To build it, run this command:

```bash
$ bazel build ...
```

That should build a binary executable that you can then run like this:

```bash
$ bazel-bin/label_image/label_image
```

##NOTE: Tensorflow and C++ will not run, as I've removed it from WORKSPACE

This uses the default example image that ships with the framework, and should
output something similar to this:

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```

In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ bazel-bin/label_image/label_image --image=my_image.png
```

For a more detailed look at this code, you can check out the C++ section of the
[Inception tutorial](https://www.tensorflow.org/tutorials/image_recognition/).

## Python implementation

label_image.py is a python implementation that provides code corresponding
to the C++ code here. This gives more intuitive mapping between C++ and
Python than the Python code mentioned in the
[Inception tutorial](https://www.tensorflow.org/tutorials/image_recognition/).
and could be easier to add visualization or debug code.


`bazel-bin/label_image/label_image_py` should be there after
```bash
$ bazel build ...
```

You can provide either image or video. If neither is provided, it will use default image.

Run

```bash
$ bazel-bin/label_image/label_image_py -v path/to/video.mp4
$ bazel-bin/label_image/label_image_py -i path/to/image.jpg
```

And get result similar to this:
```
Class: military uniform Score: 83.4305%
```
