import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa


def augmentation(img):
    img = np.array(img)
    augment = iaa.Sequential(
    [
        iaa.Fliplr(0.5),                                                # 对50%的图像进行上下翻转
        iaa.Flipud(0.5),                                                # 对50%的图像做镜像翻转

        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),
        iaa.Sometimes(0.5, iaa.Affine(                                  # 对一部分图像做仿射变换
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},                   # 图像缩放为80%到120%之间
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},     # 平移±20%之间
            rotate=(-45, 45),                                           # 旋转±45度之间
            shear=(-16, 16),                                            # 剪切变换±16度，（矩形变平行四边形）
            order=[0, 1],                                               # 使用最邻近差值或者双线性差值
            cval=(0, 255),                                              # 全白全黑填充
            mode=ia.ALL                                                 # 定义填充图像外区域的方法
        )),


        iaa.SomeOf((0, 5),
            [
                # 将部分图像进行超像素的表示
                iaa.Sometimes(0.5,
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                #锐化处理
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                #浮雕效果
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                #边缘检测，将检测到的赋值0或者255然后叠在原图上
                iaa.Sometimes(0.5,
                              iaa.OneOf([
                                        iaa.EdgeDetect(alpha=(0, 0.7)),
                                        iaa.DirectedEdgeDetect(
                                            alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # 加入高斯噪声
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # 将1%到10%的像素设置为黑色
			    # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖

                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                #5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                iaa.Invert(0.05, per_channel=True),

                # 每个像素随机加减-10到10之间的数
                iaa.Add((-10, 10), per_channel=0.5),

                # 像素乘上0.5或者1.5之间的数字.
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # 将RGB变成灰度图然后乘alpha加在原图上
                iaa.Grayscale(alpha=(0.0, 1.0)),

                #把像素移动到周围的地方
                iaa.Sometimes(0.5,
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # 扭曲图像的局部区域
                iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # 随机的顺序把这些操作用在图像上
            random_order=True
        )
    ],
        # 随机的顺序把这些操作用在图像上
        random_order=True
    )

    return Image.fromarray(augment.augment_image(img))
