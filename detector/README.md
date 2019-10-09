# Bhatkhande Character Detector

## Configuration

Install [darknet](https://github.com/AlexeyAB/darknet) to train the yolo model. The git submodule should be set up already, just `git pull --all && git submodule update` and `make` in the repo. Inference can be done using the YOLOv3 module in OpenCV.

## Labeling

Use `labelImg` wtih `labelimg/corpus` as the image directory, and `labelimg/data` as the output directory. Make sure to switch the output format to YOLO instead of PascalVOC.

## Training

Create a directory called `train_data` and move images and annotations from `labelimg/corpus` and `labelimg/data` into it. Make sure you only move the images that are annotated. Create two files, `yolo/train.txt` and `yolo/test.txt` that contain the paths (relative to this directory) to the images. This should look something like this:
```txt
train_data/corpus-027.jpg
train_data/corpus-028.jpg
...
```
Make sure that for every image above, there exists a corresponding `.txt` file. So, in the above case, `train_data/corpus-027.txt` and `train_data/corpus-028.txt` should exist. 

To start training, first download the pretrained weights for the tiny yolo model [here](https://pjreddie.com/media/files/yolov3-tiny.weights). Since these weights are for a network with a different number of classes, we need to convert them to the number of classes we have, 3. To do this, run `darknet/darknet partial yolo/tiny-yolo.cfg yolov3-tiny.weights yolov3-tiny.conv.3 3`

This will create a file called `yolov3-tiny.conv.3` that are the transfer learning weights for 3 classes. 

Create a directory called `backup` to store weight snapshots.

Finally, to actually start training, once all the files are in the right places, run:

```bash
darknet/darknet detector train yolo/data.cfg yolo/tiny-yolo.cfg yolov3-tiny.conv.3
```

As specified in the `yolo/data.cfg` file, this will create backup weight snapshots in the `backup` directory.

---

### TODO

- Tweak image size, other params in yolo/tiny-yolo.cfg
- Modify anchors using something like [this](https://github.com/pjreddie/darknet/issues/901#issuecomment-399709940)
- Try the regular yolo model instead of tiny yolo
- Annotate more images!
