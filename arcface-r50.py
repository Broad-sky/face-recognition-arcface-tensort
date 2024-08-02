import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import cv2
import torch

BATCH_SIZE = 1
INPUT_H = 112
INPUT_W = 112
OUTPUT_SIZE = 512
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-5

WEIGHT_PATH = "./backbone_iresnet50.wts"
ENGINE_PATH = "./backbone_iresnet50.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    print(count, len(lines))
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def addBatchNorm2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)


def addPrelu(network, weight_map, input, layer_name):
    relu1 = network.add_activation(input, type=trt.ActivationType.RELU)

    size = len(weight_map[layer_name+".weight"])

    # relu1.get_output(0)
    scale_11 = network.add_scale(input=input,
                      mode=trt.ScaleMode.CHANNEL,
                      shift=np.zeros((size),dtype=np.float32),
                      scale=-1*np.ones((size),dtype=np.float32))#,
                      #power=np.ones((size),dtype=np.float32))

    relu12 = network.add_activation(scale_11.get_output(0), type=trt.ActivationType.RELU)


    scale_12 = network.add_scale(input=relu12.get_output(0),
                    mode=trt.ScaleMode.CHANNEL,
                    shift=np.zeros((size),dtype=np.float32),
                    scale=-1*weight_map[layer_name+".weight"])#,
                    #power=np.ones((size),dtype=np.float32))

    ew1 = network.add_elementwise(relu1.get_output(0), scale_12.get_output(0),
                                    trt.ElementWiseOperation.SUM)
    
    return ew1


def bottleneck(network, weight_map, input, in_channels, out_channels, stride,
               layer_name, dim_match):

    bn1 = addBatchNorm2d(network, weight_map, input,
                         layer_name + "bn1", EPS)
    assert bn1

    conv1 = network.add_convolution(input=bn1.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map[layer_name +
                                                      "conv1.weight"],
                                    bias=trt.Weights())
    conv1.padding = (1, 1)
    assert conv1

    bn2 = addBatchNorm2d(network, weight_map, conv1.get_output(0),
                         layer_name + "bn2", EPS)
    assert bn2

    prelu = addPrelu(network,weight_map,bn2.get_output(0),layer_name+"prelu")
    assert prelu

    conv2 = network.add_convolution(input=prelu.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map[layer_name +
                                                      "conv2.weight"],
                                    bias=trt.Weights())
    assert conv2
    conv2.stride = (stride, stride)
    conv2.padding = (1, 1)

    bn3 = addBatchNorm2d(network, weight_map, conv2.get_output(0),
                         layer_name + "bn3", EPS)
    assert bn3

    if dim_match:
        ew1 = network.add_elementwise(input, bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    else:
        conv3 = network.add_convolution(
            input=input,
            num_output_maps=out_channels,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights())
        assert conv3
        conv3.stride = (stride, stride)

        bn4 = addBatchNorm2d(network, weight_map, conv3.get_output(0),
                             layer_name + "downsample.1", EPS)
        assert bn4

        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    assert ew1

    return ew1


def create_engine(maxBatchSize, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map["conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (1, 1)
    conv1.padding = (1, 1)

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0), "bn1", EPS)
    assert bn1

    prelu = addPrelu(network,weight_map, bn1.get_output(0),"prelu")
    assert prelu

    # pool1 = network.add_pooling(input=relu1.get_output(0),
    #                             window_size=trt.DimsHW(3, 3),
    #                             type=trt.PoolingType.MAX)
    # assert pool1
    # pool1.stride = (2, 2)
    # pool1.padding = (1, 1)

    x = bottleneck(network, weight_map, prelu.get_output(0), 64, 64, 2,
                   "layer1.0.",False)
    x = bottleneck(network, weight_map, x.get_output(0), 64, 64, 1,
                   "layer1.1.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 64, 64, 1,
                   "layer1.2.",True)

    x = bottleneck(network, weight_map, x.get_output(0), 64, 128, 2,
                   "layer2.0.",False)
    x = bottleneck(network, weight_map, x.get_output(0), 128, 128, 1,
                   "layer2.1.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 128, 128, 1,
                   "layer2.2.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 128, 128, 1,
                   "layer2.3.",True)

    x = bottleneck(network, weight_map, x.get_output(0), 128, 256, 2,
                   "layer3.0.",False)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.1.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.2.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.3.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.4.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.5.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.6.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.7.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.8.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.9.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.10.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.11.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.12.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 256, 1,
                   "layer3.13.",True)

    x = bottleneck(network, weight_map, x.get_output(0), 256, 512, 2,
                   "layer4.0.",False)
    x = bottleneck(network, weight_map, x.get_output(0), 512, 512, 1,
                   "layer4.1.",True)
    x = bottleneck(network, weight_map, x.get_output(0), 512, 512, 1,
                   "layer4.2.",True)

    bn2 = addBatchNorm2d(network, weight_map, x.get_output(0), "bn2", EPS)
    assert bn2

    fc1 = network.add_fully_connected(input=bn2.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map['fc.weight'],
                                      bias=weight_map['fc.bias'])
    assert fc1

    bn3 = addBatchNorm2d(network, weight_map, fc1.get_output(0), "features", EPS)
    assert bn3

    bn3.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(bn3.get_output(0))

    # Build engine
    builder.max_batch_size = maxBatchSize
    config.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def APIToModel(maxBatchSize):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = create_engine(maxBatchSize, builder, config, trt.float32)
    assert engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()


def compute_sim(feat1, feat2):
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

def read_image(img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python resnet50.py -s   # serialize model to plan file\n"
            "python resnet50.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        APIToModel(BATCH_SIZE)
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        # data = np.ones((BATCH_SIZE * 3 * INPUT_H * INPUT_W), dtype=np.float32)
        

        host_out0 = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)
        host_out1 = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        data0 = read_image("./face_0.jpg")
        data1 = read_image("./face_2.jpg")

        host_in0 = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W,
                                        dtype=np.float32)
        np.copyto(host_in0, data0.ravel())

        host_in1 = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W,
                                        dtype=np.float32)
        np.copyto(host_in1, data1.ravel())

        doInference(context, host_in0, host_out0, BATCH_SIZE)
        doInference(context, host_in1, host_out1, BATCH_SIZE)

        print(f'Output0: \n{host_out0[:10]}\n{host_out0[-10:]}')

        print(f'Output1: \n{host_out1[:10]}\n{host_out1[-10:]}')

        sim = compute_sim(host_out0, host_out1)
        if sim<0.2:
            conclu = 'They are NOT the same person'
        elif sim>=0.2 and sim<0.28:
            conclu = 'They are LIKELY TO be the same person'
        else:
            conclu = 'They ARE the same person'
        print(conclu)


