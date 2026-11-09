# assets/

Download the pretrained detector here as `yolov5s.onnx` (input 1x3x640x640).
https://my.sharif.edu/s/kzNyRYGxKFkrmWa

The file shipped with this lab is an **Ultralytics** export of YOLOv5s (output `1x84x8400`:
4 box values + 80 class scores for each of 8400 proposals, no separate objectness).
`detector.py` also accepts the classic YOLOv5 export (output `1x25200x85`) and YOLOv8-style
models, so you can load other `.onnx` detectors with the folder button of the *Model* box.

How to produce a model (instructor, once). Ultralytics export (anchor-free, 8400 proposals):

```
pip install ultralytics
yolo export model=yolov5su.pt format=onnx imgsz=640 opset=12
```

Alternative, classic YOLOv5 export (25200 proposals):

```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
python export.py --weights yolov5s.pt --include onnx --imgsz 640 --opset 12
```
