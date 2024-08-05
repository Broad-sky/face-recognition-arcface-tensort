# Iresnet

IResNe thttps://github.com/iduta/iresnet

## TensorRT C++ API

```
1. put arcface_backbone_iresnet50.engine into face-recognition-arcface-tensort/build

2. build and run

cd face-recognition-arcface-tensort

mkdir build

cd build

cmake ..

make

sudo ./arcface-r50-sps -s   // serialize model to plan file i.e. 'arcface_backbone_iresnet50.engine'
sudo ./arcface-r50-sps -d   // deserialize plan file and run inference

```

### TensorRT Python API

```
python arcface-r50.py -s   // serialize model to plan file i.e. 'arcface_backbone_iresnet50.engine'
python arcface-r50.py -d   // deserialize plan file and run inference
```

### Result

![屏幕截图 2024-08-05 131800](C:\Users\sps\Desktop\屏幕截图 2024-08-05 131800.png)
