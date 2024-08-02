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

sudo ./arcface-r50-sps -s   // serialize model to plan file i.e. 'resnet18.engine'
sudo ./arcface-r50-sps -d   // deserialize plan file and run inference

```

### TensorRT Python API

```
python arcface-r50.py -s   // serialize model to plan file i.e. 'wide_resnet50.engine'
python arcface-r50.py -d   // deserialize plan file and run inference
```

### Result

![img](https://user-images.githubusercontent.com/15235574/83122953-f45f8d80-a106-11ea-84b0-4f6ff91b5924.jpg)

