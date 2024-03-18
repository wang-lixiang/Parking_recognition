from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob

from keras.models import load_model

from Parking import Parking
import pickle

cwd = os.getcwd()


# 返回的是我们识别到的每个车位的坐标点形成的字典
def img_process(test_images, park):
    # Step 1: 提取白色和黄色区域的图像
    # map 函数将一个函数应用到一个可迭代对象的每个元素上，对每个图像都进行同样的操作
    # list函数将每个结果存储在一个结果列表中
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)

    # Step 2: 将图像转换为灰度
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    # Step 3: 检测图像边缘
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)

    # Step 4: 选择停车场大致区域
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # Step 5: 应用霍夫变换检测直线
    list_of_lines = list(map(park.hough_lines, roi_images))

    # Step 6: 在原始图像上绘制检测到的直线
    line_images = []
    # 将两个可迭代对象 test_images 和 list_of_lines 中对应位置的元素打包成元组，并返回一个由这些元组组成的迭代器。
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines))
    park.show_images(line_images)

    # Step 7: 识别停车位的边界矩形
    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)

    park.show_images(rect_images)

    # 为每个图像绘制停车位，并将结果保存在列表中
    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    # 显示绘制停车位后的图像
    park.show_images(delineated)
    # 经过对比，我们发现第二幅图的提取效果比较好，选择第二个图像的停车位信息
    final_spot_dict = spot_pos[1]
    # 打印停车位数量
    print(len(final_spot_dict))

    # 将最终的停车位信息保存到文件中
    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict


# 导入我们训练好的模型
def keras_model(weights_path):
    model = load_model(weights_path)
    return model


# 测试图像的效果
def img_test(test_images, final_spot_dict, model, class_dictionary):
    for i in range(len(test_images)):
        predicted_images = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)


# 测试视频的效果
def video_test(video_name, final_spot_dict, model, class_dictionary):
    park.predict_on_video(video_name, final_spot_dict, model, class_dictionary, ret=True)


if __name__ == '__main__':
    # 识别测试的图片，这里将会形成一个列表2* 720*1280*3
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    weights_path = 'car1.h5'
    video_name = 'parking_video.mp4'

    # 模型识别车位的情况返回的是0，1，需要一个字典指定是空还是占据着
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'

    park = Parking()
    park.show_images(test_images)
    final_spot_dict = img_process(test_images, park)
    # 加载已经训练好的模型
    model = keras_model(weights_path)

    # 图片测试
    # img_test(test_images, final_spot_dict, model, class_dictionary)
    # 视频测试，不过很吃电脑配置
    video_test(video_name, final_spot_dict, model, class_dictionary)
