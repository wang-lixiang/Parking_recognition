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

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    def filter_region(self, image, vertices):
        """
                剔除掉不需要的地方
        """
        mask = np.zeros_like(image)
        if len(mask.shape) == 2:
            cv2.fillPoly(mask, vertices, 255)
            self.cv_show('mask', mask)
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

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5
        spot_dict = {}  # 字典：一个车位对应一个位置
        tot_spots = 0
        # 微调
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            num_splits = int(abs(y2 - y1) // gap)
            for i in range(0, num_splits + 1):
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if key > 0 and key < len(rects) - 1:
                # 竖直线
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 计算数量
            if key == 0 or key == (len(rects) - 1):
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # 字典对应好
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
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        return new_image, spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return new_image

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁剪
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            spot_id = spot_dict[spot]

            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))

            cv2.imwrite(os.path.join(folder_name, filename), spot_img)

    def make_prediction(self, image, model, class_dictionary):
        # 预处理
        img = image / 255.

        # 转换成4D tensor
        image = np.expand_dims(img, axis=0)

        # 用训练好的模型进行训练
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        label = class_dictionary[inID]
        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():
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

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
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
