# -*- coding:utf8 -*-
import sys
import paddle.v2 as paddle
from PIL import Image
import numpy as np
import os

reload(sys)
sys.setdefaultencoding('utf8')
paddle.init(use_gpu=False, trainer_count=1)
datadim = 3 * 32 * 32
classdim = 10

image = paddle.layer.data(
    name="image", type=paddle.data_type.dense_vector(datadim))


def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(
        input=fc1,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
    return fc2


net = vgg_bn_drop(image)
out = paddle.layer.fc(input=net,
                      size=classdim,
                      act=paddle.activation.Softmax())


def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    # PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。
    # PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
    im = im.transpose((2, 0, 1))  # CHW
    # CIFAR训练图片通道顺序为B(蓝),G(绿),R(红),
    # 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
    im = im[(2, 1, 0), :, :]  # BGR
    im = im.flatten()
    im = im / 255.0
    return im


def readFile(file):
    fs = open(file)


def endover(imageNames):
    test_data = []
    cur_dir = os.getcwd()
    imageName = "media/" + imageNames
    passName = "/usr/pass/2/params_pass_199.tar"
    labbesl = np.array(["0:土豆丝", "1:清蒸鱼", "2:松仁玉米", "3:西红柿炒鸡蛋", "4:炒西兰花", "5:虾仁", "6:包子", "7:皮蛋", "8:炒花生米", "9:韭菜鸡蛋"])
    test_data.append((load_image(imageName),))

    with open(passName, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    probs = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    # print probs
    print lab
    labelName = []
    labelNum = []
    for i in range(5):
        labelName.append(labbesl[lab[0][i]])
        labelNum.append(lab[0][i])
    value = {}
    value['date'] = labelNum
    value['name'] = labelName
    print labelName
    return value

    # print "Label of %s is: %d" % (imageName, lab[0][0])
