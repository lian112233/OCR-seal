import math
import os
import cv2
# import easyocr
import numpy as np
import torch
from PIL import Image
from ELSDC_extract_seal import resize_img, erase_black_pixel, preprocess_, \
    eldsc_extract, copy_out, make_out_folder, angle_diff, IMG2PGM
import argparse

import pandas as pd
from paddleocr import PaddleOCR, draw_ocr

# from yolov5.utils.datasets import LoadImages
# from yolov5.utils.torch_utils import select_device, load_classifier, time_sync

repo = 'yolov5'
model_path = 'yolov5/runs/train/exp29/weights/best.pt'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
MIN_RADIAN = 3.14 * 0.25
MIN_RADIUS_RATIO = 0.8
MIN_RADIUES = 55.0
AXBX = 0.9
OUT_ELLIPSE = r'./out_ellipse.txt'
OUT_POLYGON = './out_polygon.txt'
OUT_SVG = r'./output.svg'


# class Circle:
#     radius = 0
#     circum
def main(opt):
    run(**vars(opt))


def distance_two_points(idx1, idx2):
    dx = idx1[0] - idx2[0]
    dy = idx1[1] - idx2[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance


def is_chinese(text):
    ch_num = 0
    for ch in text:
        if '\u4e00' < ch < '\u9fff':
            ch_num += 1
    return ch_num


def getDist_P2L(PointP, Pointa, Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程

    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]
    distance = (A * PointP[0] + B * PointP[1] + C) / math.sqrt(A * A + B * B)

    return distance


def scale(data, sec_dis):
    """多边形等距缩放
    Args:
        data: 多边形按照逆时针顺序排列的的点集
        sec_dis: 缩放距离

    Returns:
        缩放后的多边形点集
    """
    num = len(data)
    scal_data = []
    for i in range(num):
        x1 = data[(i) % num][0] - data[(i - 1) % num][0]
        y1 = data[(i) % num][1] - data[(i - 1) % num][1]
        x2 = data[(i + 1) % num][0] - data[(i) % num][0]
        y2 = data[(i + 1) % num][1] - data[(i) % num][1]

        d_A = (x1 ** 2 + y1 ** 2) ** 0.5
        d_B = (x2 ** 2 + y2 ** 2) ** 0.5

        Vec_Cross = (x1 * y2) - (x2 * y1)
        if (d_A * d_B == 0):
            continue
        sin_theta = Vec_Cross / (d_A * d_B)
        if (sin_theta == 0):
            continue
        dv = sec_dis / sin_theta

        v1_x = (dv / d_A) * x1
        v1_y = (dv / d_A) * y1

        v2_x = (dv / d_B) * x2
        v2_y = (dv / d_B) * y2

        PQ_x = v1_x - v2_x
        PQ_y = v1_y - v2_y

        Q_x = data[(i) % num][0] + PQ_x
        Q_y = data[(i) % num][1] + PQ_y
        scal_data.append([Q_x, Q_y])
    return scal_data


def getDist_P2P(Pointa, Pointb):
    """计算点到点的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程

    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    # 代入点到直线距离公式

    distance = math.sqrt(A * A + B * B)

    return distance


def perspective(image, pts1):
    w, h = image.shape[:2]
    pts_max = w if w < h else h
    pts1 = np.float32(pts1.reshape((4, 2)))
    # 原图中的点的坐标 四个
    # pts1 = np.float32([[56, 65], [250, 52], [28, 200], [280, 290]])
    # 变换到新图片中，四个点对应的新的坐标 一一对应
    pts2 = np.float32([[0, 0], [0, pts_max], [pts_max, pts_max], [pts_max, 0]])

    # 生成变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(image, M, (image.shape[0], image.shape[1]))

    return dst


def perspective_trans(img):
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(img, element, iterations=1)
    edges = cv2.Canny(erosion, 32, 128)
    contours, hierarchy = cv2.findContours(edges, 3, 2)
    area = map(cv2.contourArea, contours)
    area = list(area)
    area_idx = area.index(max(area))
    cnt = contours[area_idx]
    # 2.进行多边形逼近，得到多边形的角点
    approx1 = cv2.approxPolyDP(cnt, 75, True)
    approx2 = cv2.approxPolyDP(cnt, 15, True)
    # approx1_reshape = np.array(approx1).reshape((4, 2))
    approx2_reshape = np.array(approx2).reshape((approx2.shape[0], approx2.shape[2]))
    dis_app2_1_02 = getDist_P2L(approx2_reshape[1], approx2_reshape[0], approx2_reshape[2])
    dis_app2_3_24 = getDist_P2L(approx2_reshape[3], approx2_reshape[4], approx2_reshape[2])
    dis_app2_5_46 = getDist_P2L(approx2_reshape[5], approx2_reshape[4], approx2_reshape[6])
    dis_app2_7_60 = getDist_P2L(approx2_reshape[7], approx2_reshape[6], approx2_reshape[0])
    approx1 = [approx2_reshape[1], approx2_reshape[3], approx2_reshape[5], approx2_reshape[7]]
    max_p2l = max(np.fabs((dis_app2_1_02, dis_app2_3_24, dis_app2_5_46, dis_app2_7_60)))
    # dis_app1_0_2 = getDist_P2P(approx1_reshape[0], approx1_reshape[2])  # approx1点[0]到点[2]距离
    # dis_app1_1_3 = getDist_P2P(approx1_reshape[1], approx1_reshape[3])

    # scale_pro = dis_app1_0_2 * 0.2 if dis_app1_0_2 * 0.2 > dis_app1_1_3 * 0.2 else dis_app1_1_3 * 0.2
    data1 = scale(approx1, -max_p2l)
    img_copy = img.copy()
    # cv2.polylines(img_copy, [approx1], True, (255, 0, 0), 2)
    cv2.polylines(img_copy, [approx2], True, (0, 255, 0), 2)
    cv2.polylines(img_copy, [np.int32(np.array(data1).reshape((4, 1, 2)))], True, (0, 255, 255), 2)
    dst = perspective(img, np.array(data1))
    return dst, img_copy


def select_result(roi, img_name, stamp_idx, save_flag):
    roi_w, roi_h = roi.shape[1], roi.shape[0]
    radius = roi_w / 2 if roi_w < roi_h else roi_h / 2
    radius = radius * 0.95
    trans_center = (roi_w / 2, roi_h / 2)
    circles = find_circle(img_name + '.png', roi, './eldsc_out', stamp_idx)
    result = polar_and_ocr(roi, radius, trans_center, img_name + str(stamp_idx))
    results = []

    if result != -1:
        result_dict = {"result": result, "img_name": img_name + str(stamp_idx), "circle": (trans_center, radius)}
        results.append(result_dict)
    if circles != -1:

        for cir_idx, circle in enumerate(circles, 0):
            radius = int(circle['a']) if int(circle['a']) >= int(circle['b']) else int(circle['b'])
            trans_center = (int(circle['x_c']), int(circle['y_c']))
            result = polar_and_ocr(roi, radius, trans_center,
                                   img_name + str(stamp_idx) + "cir_idx" + str(cir_idx))
            if result != -1:
                result_dict = {"result": result, "img_name": img_name + str(stamp_idx) + "cir_idx" + str(cir_idx),
                               "circle": (trans_center, radius)}
                results.append(result_dict)
    ch_num_list = []
    avg_score_list = []
    if len(results) == 0:
        return -1
    for res_idx, result in enumerate(results, 0):

        txts = [line[1][0] for line in result["result"]]
        scores = [line[1][1] for line in result["result"]]
        ch_num = 0
        sum_score = 0
        be_skip = 0

        for t_idx, txt in enumerate(txts, 0):

            if is_chinese(txt) == 0:
                be_skip += 1
                continue

            sum_score += scores[t_idx]
            ch_num += is_chinese(txt)
        avg_score = sum_score / (len(txts) - be_skip + 1)  # 防止除数为0
        avg_score_list.append((avg_score, res_idx))
        ch_num_list.append((ch_num, res_idx))
    max_ch_num = max(ch_num_list, key=lambda x: x[0])
    out_result = None
    highest_score = 0
    result_idx = 0

    for idx, item in enumerate(ch_num_list, 0):

        if item == max_ch_num:
            if avg_score_list[idx][0] > highest_score:
                highest_score = avg_score_list[idx][0]
                result_idx = item[1]
                print(str(results[item[1]]) + "!!")
                out_result = results[item[1]]
    if out_result is None:
        return -1
    if save_flag is True:
        polarImg_path = 'Stamp_res/polarImg' + out_result["img_name"] + '.png'
        image = Image.open(polarImg_path).convert('RGB')
        boxes = [line[0] for line in out_result["result"]]
        txts = [line[1][0] for line in out_result["result"]]
        scores = [line[1][1] for line in out_result["result"]]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
        im_show = Image.fromarray(im_show)
        result_path = 'Stamp_res_ocr/res' + img_name + str(stamp_idx) + '.png'
        im_show.save(result_path)

    return out_result


def select_ch_result(result):
    boxes = [line[0] for line in result["result"]]
    txts = [line[1][0] for line in result["result"]]
    sorted_result = []
    for idx, txt in enumerate(txts, 0):
        if is_chinese(txt) > 0:
            sorted_result.append((boxes[idx], txt))
    sorted_result.sort(key=lambda x: x[0][0])
    return sorted_result


#  旋转图像
def rotate_target(img: np.array, result):
    w, h = img.shape[:2]
    sort_result = select_ch_result(result)
    boxes = [line[0] for line in sort_result]
    txts = [line[1] for line in sort_result]
    first_box = boxes[0]
    last_box = boxes[-1]
    single_ch_width = (last_box[1][0] - last_box[0][0]) / len(txts[-1])

    if first_box != last_box:

        if is_chinese(txts[0]) > 0 and is_chinese(txts[-1]) > 0:
            rotate_arc_length = (last_box[1][0] - last_box[0][0]) * 1.2
            circumference = result["circle"][1] * 2 * math.pi
            angle = rotate_arc_length / circumference * 360
            center = result["circle"][0]
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            rotated_img = cv2.warpAffine(img, rotate_matrix, (w, h))
            return rotated_img
        else:
            return img

    elif first_box[0][0] <= (single_ch_width * 0.8):
        rotate_arc_length = single_ch_width * 1.1
        circumference = result["circle"][1] * 2 * math.pi
        angle = rotate_arc_length / circumference * 360
        center = result["circle"][0]
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        rotated_img = cv2.warpAffine(img, rotate_matrix, (w, h))
        return rotated_img
    else:
        return img


def find_circle(filename, img, output_path, stamp_idx):
    output_path = make_out_folder(output_path, filename)
    # dst, img_copy = perspective_trans(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(gray_img, element1, iterations=1)
    new_name = filename[:filename.find('.')] + str(stamp_idx)
    pgm_path = fr"{os.path.join(output_path, new_name + '.pgm')}"
    img_path = fr"{os.path.join(output_path, new_name + '.png')}"
    dst_path = fr"{os.path.join(output_path, new_name + 'dst.png')}"
    img_copy_path = fr"{os.path.join(output_path, new_name + 'copy.png')}"
    cv2.imwrite(pgm_path, erosion)
    cv2.imwrite(img_path, img)
    # cv2.imwrite(dst_path, dst)
    # cv2.imwrite(img_copy_path, img_copy)
    print(os.system("./elsdc " + pgm_path))
    out_ellipse_path = 'out_ellipse.txt'
    copy_out([OUT_ELLIPSE, OUT_SVG], output_path)

    no_stamp_flag = 0

    with open(out_ellipse_path) as out_ellipse:
        ellipses = out_ellipse.readlines()

    ellipses_data = []
    ellipse_data_type = ["ellipse_id", "x1", "y1", "x2", "y2", "x_c", "y_c", "a", "b",
                         "theta", "ang_start", "ang_end"]

    for num, ellipse in enumerate(ellipses, 0):
        ellipse_list = ellipse.split()
        ellipse_list = [float(x) for x in ellipse_list]
        ellipse_data = dict(zip(ellipse_data_type, ellipse_list))
        ellipses_data.append(ellipse_data)
        # 计算圆弧的弧度
        temp = angle_diff(float(ellipse_data['ang_start']), float(ellipse_data['ang_end']))
        ring_angle = float(ellipse_data['ang_start']) - temp

        if ring_angle < -math.pi:
            ring_angle = ring_angle + 2 * math.pi
        if ring_angle == float(ellipse_data['ang_end']):
            ellipse_data.setdefault('radians', 2 * math.pi - temp)
        else:
            ellipse_data.setdefault('radians', temp)

        ellipses_data[num] = ellipse_data

    max_radius = 0

    for num, ellipse in enumerate(ellipses_data, 0):

        if ellipse['radians'] > MIN_RADIAN and img.shape[0] / 2 > ellipse['a'] > max_radius:
            print(ellipse, MIN_RADIAN)
            max_radius = ellipse['a']

    temp_count = 0
    arcs_length = list()
    selected_ellipse = []

    for num, ellipse_data in enumerate(ellipses_data, 0):

        if ellipse_data['radians'] > MIN_RADIAN and float(ellipse_data['a']) > (MIN_RADIUS_RATIO * max_radius) \
                and float(ellipse_data['a']) / float(ellipse_data['b']) >= AXBX and \
                distance_two_points((ellipse_data['x_c'], ellipse_data['y_c']), (img.shape[0] / 2, img.shape[1] / 2)) < \
                img.shape[0] / 15:
            # 算弧长
            arc_length = ellipse_data['radians'] * ellipse_data['a']
            area = math.pi * ellipse_data['a'] * ellipse_data['b']
            ellipse_data.setdefault('arc_length', arc_length)
            ellipse_data.setdefault('area', area)
            arcs_length.append(arc_length)
            selected_ellipse.append(ellipse_data)
            temp_count += 1

    if len(selected_ellipse) != 0:
        return selected_ellipse
    else:
        return -1


def run(weights=model_path,
        source='train',
        repo=repo,
        img_size=640):
    model = torch.hub.load(repo, 'custom', path=weights,
                           source='local')  # local repo
    files = []

    if os.path.isdir(source):
        files = sorted([os.path.join(source, x) for x in os.listdir(source)])  # dir
    elif os.path.isfile(source):
        files = [source]

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for path in images:
        print("Current pic: " + path)
        img = resize_img(cv2.imread(path), img_size)
        img_name = path.split('/')[-1].split('.')[0]
        result = model(img)
        result_pd = result.pandas()
        xywh = result_pd.xywh[0]
        xyxy = result_pd.xyxy[0]
        xmins = xyxy['xmin'].astype(int)
        ymins = xyxy['ymin'].astype(int)
        xmaxs = xyxy['xmax'].astype(int)
        ymaxs = xyxy['ymax'].astype(int)

        for idx, dets in enumerate(xywh['xcenter'], 0):
            x_min, y_min, x_max, y_max = xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx]
            ROI = img[y_min:y_max, x_min:x_max].copy()
            # ROI = perspective_trans(ROI)
            ROI = erase_black_pixel(ROI)
            ROI = resize_img(ROI, 200)
            # roi_w, roi_h = ROI.shape[1], ROI.shape[0]
            # radius = roi_w / 2 if roi_w < roi_h else roi_h / 2
            # radius = radius * 0.95
            # trans_center = (roi_w / 2, roi_h / 2)
            # circles = find_circle(img_name + str(idx) + '.png', ROI, './eldsc_out')
            # gray_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            # results = [polar_and_ocr(ROI, radius, trans_center, img_name + str(idx))]
            #
            # if circles != -1:
            #
            #     for cir_idx, circle in enumerate(circles, 0):
            #         radius = int(circle['a']) if int(circle['a']) >= int(circle['b']) else int(circle['b'])
            #         trans_center = (int(circle['x_c']), int(circle['y_c']))
            #         results.append(polar_and_ocr(ROI, radius, trans_center,
            #                                      img_name + str(idx) + "cir_idx" + str(cir_idx)))
            out_result = select_result(ROI, img_name, idx, False)
            if out_result != -1:
                rotated_img = rotate_target(ROI, out_result)
                out_result = select_result(rotated_img, img_name, idx, True)
            else:
                print(path + " have no result!")


def polar_and_ocr(ROI, radius, trans_center, img_name):
    circumference = 2 * math.pi * radius
    ROI_trans = cv2.transpose(ROI)
    ROI = cv2.flip(ROI_trans, 0)
    polarImg = cv2.warpPolar(ROI, (int(radius), int(circumference)), trans_center, radius,
                             cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    polarImg = cv2.flip(polarImg, 1)  # 镜像
    polarImg = cv2.transpose(polarImg)  # 转置
    # cv2.imshow('polarImg',polarImg)
    # cv2.waitKey(0)
    # polarImg = resize_img(polarImg, 640)
    polarImg_path = 'Stamp_res/polarImg' + img_name + '.png'
    cv2.imwrite(polarImg_path, polarImg)

    # OCR识别-
    # reader = easyocr.Reader(['ch_sim', 'en'])
    # result = reader.readtext('Stamp_res/polarImg' + img_name+str(idx) + '.png', polarImg)
    ocr = PaddleOCR(use_angle_cls=True,
                    lang="ch")  # need to run only once to download and load model into memory
    img_path = polarImg_path
    result = ocr.ocr(img_path, cls=True)

    if len(result) == 0:
        return -1

    for line in result:
        print(line)

    return result


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=model_path,
                        help='model.pt path(s)')
    parser.add_argument('--repo', type=str, default=repo,
                        help='yolov5-dir-path')
    parser.add_argument('--source', type=str,
                        default='./train',
                        help='image path or images-dir path')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = args_parser()
    main(opt)
