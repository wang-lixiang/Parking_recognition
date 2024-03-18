import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


class Parking:

    def show_images(self, images, cmap=None):
        # 计算所需的行数和列数，以便以网格布局显示图像
        cols = 2
        rows = (len(images) + 1) // cols

        # 创建一个新的图形，设置其大小为(15, 12)
        plt.figure(figsize=(15, 12))
        # 遍历图像列表中的每个图像
        for i, image in enumerate(images):
            # 创建一个子图，放置在网格中的相应位置
            plt.subplot(rows, cols, i + 1)
            # 检查图像是否为灰度图（2D）或彩色图（3D），并设置相应的颜色映射
            cmap = 'gray' if len(image.shape) == 2 else cmap
            # 显示图像，并使用指定的颜色映射
            plt.imshow(image, cmap=cmap)
            # 移除坐标轴上的刻度标记
            plt.xticks([])
            plt.yticks([])
        # 调整子图的布局，以更好地适应
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # 显示图形
        plt.show()

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 从图像中选择白色和黄色的区域
    def select_rgb_white_yellow(self, image):
        # 过滤掉背景
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])

        # lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
        # cv2.inRange 函数创建一个二值掩码，用于过滤掉不在颜色范围内的部分
        white_mask = cv2.inRange(image, lower, upper)
        self.cv_show('white_mask', white_mask)
        # 使用掩码将原始图像中不符合颜色范围的部分置为黑色，以突出白色和黄色部分
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        self.cv_show('masked', masked)

        return masked

    # 将图像转化为灰度图
    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 边缘检测
    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    # 过滤掉图像中不需要的地方，保留指点区域
    def filter_region(self, image, vertices):
        # 创建一个和原图大小相同的全零数组
        mask = np.zeros_like(image)
        if len(mask.shape) == 2:
            # 将指定的vertices多边形区域填充为白色
            cv2.fillPoly(mask, vertices, 255)
            self.cv_show('mask', mask)
        # 将原图和mask进行按位与操作，保留mask区域内原图像像素
        return cv2.bitwise_and(image, mask)

    # 通过手动定义几个点，画出一个大致的多边形，框出停车场大致的轮廓
    def select_region(self, image):
        # cols=1280, rows=720
        rows, cols = image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]

        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
        for point in vertices[0]:
            # 在指定的点画一个圆
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    # 检测图像中想要的直线
    def hough_lines(self, image):
        # 输入的图像需要是边缘检测后的结果
        # minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
        # rho距离精度,theta角度精度,threshod超过设定阈值才被检测出线段
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    # 在图像中画出识别出来的停车位线
    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # 过滤霍夫变换检测到直线
        # 创建输入图像的副本，避免修改原始图像
        if make_copy:
            image = np.copy(image)
        # 创建一个空列表，用于存储经过滤波后的直线。
        cleaned = []
        for line in lines:
            # 解包直线的起点和终点坐标。
            for x1, y1, x2, y2 in line:
                # 根据直线的长度和只想的斜率过滤这些直线
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))
                    # 在图像上绘制直线，使用指定的颜色和粗细
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        print("滤波后检测到的直线数量: ", len(cleaned))
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        # Step 1: 过滤部分直线
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        # Step 2: 对直线按照x1进行排序
        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

        # Step 3: 找到多个列，相当于每列是一排车
        clusters = {}
        dIndex = 0
        clus_dist = 10

        for i in range(len(list1) - 1):
            distance = abs(list1[i + 1][0] - list1[i][0])
            if distance <= clus_dist:
                if not dIndex in clusters.keys(): clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])

            else:
                dIndex += 1

        # Step 4: 得到坐标
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1

        print("Num Parking Lanes: ", len(rects))
        # Step 5: 把列矩形画出来
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    # 在图像上绘制停车位标记
    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        # 如果make_copy为True，则创建图像的副本以免修改原始图像
        if make_copy:
            new_image = np.copy(image)
        # 停车位的间隔是固定的
        gap = 15.5
        # 用于存储车位位置信息的字典
        spot_dict = {}
        # 初始化总停车位数
        tot_spots = 0
        # 微调每个矩形的位置，这里给出了微调的偏移量
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}
        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}

        # 遍历每个矩形
        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            # 绘制每列停车位的大矩形
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 计算停车位间的水平线数量
            num_splits = int(abs(y2 - y1) // gap)
            # 在大矩形中绘制水平线
            for i in range(0, num_splits + 1):
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            # 在非边缘位置绘制竖直线
            if key > 0 and key < len(rects) - 1:
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 计算总停车位数
            if key == 0 or key == (len(rects) - 1):
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # 将每个车位与其编号对应起来
            if key == 0 or key == (len(rects) - 1):
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            else:
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)
        # 如果save为True，则保存绘制了停车位的图像
        if save:
            filename = 'with_parking.jpg'
            # 两个参数：文件路径和要保存的图像
            cv2.imwrite(filename, new_image)
        # 返回新图像和车位字典
        return new_image, spot_dict

    # 裁剪图像，将识别到的车位每个都以单独的图像显示，方便用于CNN训练
    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            # 获取车位的位置信息
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁剪车位图像
            spot_img = image[y1:y2, x1:x2]
            # 将车位图像调整大小为CNN所需的尺寸（这里将尺寸放大了2倍）
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            # 获取车位编号
            spot_id = spot_dict[spot]

            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))

            cv2.imwrite(os.path.join(folder_name, filename), spot_img)

    # 对输入图像进行分类预测，并返回预测的类别标签
    def make_prediction(self, image, model, class_dictionary):
        # 预处理：将图像像素值缩放到[0,1]范围内
        img = image / 255.
        # 将图像转换成4D张量，以符合模型的输入要求
        image = np.expand_dims(img, axis=0)
        # 使用给定的模型进行图像分类预测
        class_predicted = model.predict(image)
        # 获取预测结果中概率最高的类别索引
        inID = np.argmax(class_predicted[0])
        # 根据类别索引从类别字典中获取类别标签
        label = class_dictionary[inID]
        return label

    # 预测图像
    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        # 复制一下图像，避免修改原图
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0
        all_spots = 0

        # 遍历每个车位
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 提取车位图像并调整大小以符合模型的输入要求
            spot_img = image[y1:y2, x1:x2]
            # 48*48已经是VGG16接受的最小图像大小
            spot_img = cv2.resize(spot_img, (48, 48))

            # 如果预测为空闲车位，则在覆盖层上绘制矩形标记
            label = self.make_prediction(spot_img, model, class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1
        # 将覆盖层与原始图像叠加，产生标记效果
        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        # 在图像上添加文字信息：可用车位数量和总车位数量
        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # 如果需要保存图像，则保存，这里选择不保存
        save = False
        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            # 每五帧后处理一次
            if count == 5:
                count = 0

                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5
                for spot in final_spot_dict.keys():
                    all_spots += 1
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    spot_img = image[y1:y2, x1:x2]
                    spot_img = cv2.resize(spot_img, (48, 48))

                    label = self.make_prediction(spot_img, model, class_dictionary)
                    if label == 'empty':
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                # 在图像上添加文字信息：可用车位数量和总车位数量
                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('frame', new_image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        cap.release()
