import io
import math
import os
import cv2
import shutil
import numpy as np
from matplotlib import pyplot as plt
from typing.io import TextIO

HEIGHT_RING = 800
WIDTH_RING = 800
HEIGHT_OUT = 640
WIDTH_OUT = 640
INPUT_DIR = r'./PGM'
IMG_INPUT = r'./Stamp_dataset'
# out_ellipse = r'./out_ellipse.txt'
# os.system("./elsdc "+os.path.join(input_dir, "3.pgm"))
OUT = r'./output'
OUT_ELLIPSE = r'./out_ellipse.txt'
OUT_POLYGON = './out_polygon.txt'
OUT_SVG = r'./output.svg'
CUT = r'./CutIMG'
# file_list = os.listdir(IMG_INPUT)

MIN_RADIAN = 3.14 * 0.15
MIN_RADIUS_RATIO = 0.5
MIN_RADIUES = 55.0
AXBX = 0.7
lower_red = np.array([156, 43, 46])
upper_red = np.array([180, 255, 255])
red1_range = (lower_red, upper_red)
lower_red_2 = np.array([0, 43, 46])
upper_red_2 = np.array([12, 255, 255])
red2_range = (lower_red_2, upper_red_2)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 46])
black_range = (lower_black, upper_black)


def pixel_color_in_range(pixel_color, color_range):
    range_lower = color_range[0]
    range_upper = color_range[1]
    flag = True
    if False in (pixel_color - range_lower > 0) or False in (range_upper - pixel_color < 0):
        flag = False
    return flag


# 在黑色像素附近寻找像素颜色中值并赋值
def find_nearest_color(pixel_coordinates, img, radius, mask_black):
    global black_range
    img_shape = img.shape
    img_w = img_shape[1]
    img_h = img_shape[0]
    x = pixel_coordinates[1]
    y = pixel_coordinates[0]
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_black = cv2.dilate(mask_black, element1, iterations=1)

    start_x = x - radius if x - radius >= 0 else 0
    start_y = y - radius if y - radius >= 0 else 0
    end_x = x + radius if x + radius <= img_w else img_w
    end_y = y + radius if y + radius <= img_h else img_h
    other_color = cv2.copyTo(img, ~erosion_black)
    max_range = other_color[start_y:end_y, start_x:end_x]
    color_list = [tuple(j) for i in max_range for j in i if tuple(j) != (0, 0, 0)]

    if len(color_list) != 0:
        color_array = np.array(color_list)
        median_color = np.median(color_array, axis=0)
        return median_color
    else:
        return find_nearest_color(pixel_coordinates, img, radius + 1, mask_black)


def partial_median_red(img_hsv, pixel_coordinates, radius, mask_red, mask_black):
    global red1_range, red2_range, black_range

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_black = cv2.dilate(mask_black, element1, iterations=1)
    mask = mask_red - erosion_black
    x = pixel_coordinates[1]
    y = pixel_coordinates[0]
    img_shape = img_hsv.shape
    img_w = img_shape[1]
    img_h = img_shape[0]
    start_x = x - radius if x - radius >= 0 else 0
    start_y = y - radius if y - radius >= 0 else 0
    end_x = x + radius if x + radius <= img_w else img_w
    end_y = y + radius if y + radius <= img_h else img_h
    red = cv2.copyTo(img_hsv, mask)
    max_range = red[start_y:end_y, start_x:end_x]
    red_list = [tuple(j) for i in max_range for j in i if tuple(j) != (0, 0, 0)]

    if len(red_list) != 0:
        red_array = np.array(red_list)
        # cv2.imshow('10', red)
        max_red = np.median(red_array, axis=0)
        return max_red
    else:
        return partial_median_red(img_hsv, pixel_coordinates, radius + 1, mask_red, mask_black)


def erase_black_pixel(img):
    global black_range
    global upper_black
    img = resize_img(img, HEIGHT_OUT)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_cp = img_hsv.copy()
    # img_hsv_cp[..., 2] = 255

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    kernel = np.ones((3, 3), np.uint8)
    pre_mask_black = cv2.inRange(img_hsv, black_range[0], black_range[1])
    mask_red_1 = cv2.inRange(img_hsv_cp, lower_red, upper_red)
    mask_red_2 = cv2.inRange(img_hsv_cp, lower_red_2, upper_red_2)
    mask_red = (mask_red_1 | mask_red_2)
    pre_mask_red = mask_red & (~pre_mask_black)
    # img_show(mask_red)
    red_1 = cv2.copyTo(img_hsv, pre_mask_red)
    red_list = [tuple(j) for i in red_1 for j in i if tuple(j) != (0, 0, 0)]
    red_array = np.array(red_list)
    if len(red_array) == 0:
        return img
    min_light = np.min(red_array[:, 2]) * 1.1
    min_light = int(min_light) if min_light < 100 else 100

    upper_black[2] = min_light
    black_range = [lower_black, upper_black]
    mask = cv2.inRange(img_hsv, black_range[0], black_range[1])

    median_red = np.median(red_array, axis=0)
    real_mask_red = mask_red & (~mask)
    close_mask_red = cv2.morphologyEx(real_mask_red, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_black = mask & (~close_mask_red)
    # erosion_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=1)
    erosion_black = cv2.dilate(mask_black, element1, iterations=1)
    # img_black_copy = cv2.bitwise_or(img_hsv, erosion_black)
    # img_hsv_copy = img_hsv.copy()
    # img_show(erosion_black)
    # for i in range(mask_black.shape[1]):
    #     for j in range(mask_black.shape[0]):
    #         if mask_black[j][i] != 0:
    #             if mask_red[j][i] != 0:
    #                 img_hsv[j, i] = partial_median_red(img_hsv_copy, (j, i), 1, real_mask_red, mask_black)
    #                 # continue
    #                 # img_hsv[j, i] = median_red
    #             else:
    #                 img_hsv[j, i] = find_nearest_color((j, i), img_hsv_copy, 7, mask_black)

    out_img = cv2.inpaint(img, erosion_black, 10, cv2.INPAINT_TELEA)

    # out_img = resize_img(out_img, HEIGHT_OUT)
    # opening = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return out_img


def img_show(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(1000)


#  给每个图像创造输出文件夹
def make_out_folder(output_path: str, filename: str):
    path = os.path.join(output_path, filename.split('.')[0])
    if not os.path.exists(path):
        os.mkdir(path)
    return path


#  拷贝输出文件到对应的文件路径下
def copy_out(src_list, out):
    for file in src_list:
        shutil.copyfile(file, os.path.join(out, file))


def angle_diff(a, b):
    a -= b
    while a <= -math.pi:
        a += math.pi * 2
    while a > math.pi:
        a -= math.pi * 2
    if a < 0.0:
        a = -a
    return a


def preprocess_(img_name, img_path, output_path):
    img = cv2.imread(os.path.join(img_path, img_name))
    # out = resize_img(img, HEIGHT_RING)

    # 3. 膨胀和腐蚀操作的核函数

    bmp_path = fr"{os.path.join(output_path, img_name)}"
    cv2.imwrite(bmp_path, img)
    pgm_path = IMG2PGM(img_name, output_path, output_path)
    return pgm_path, bmp_path


def LAB_extract(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_shape = img.shape
    img_w = img_shape[1]
    img_h = img_shape[0]
    post_list = list()
    final_list = list()
    img_as = lab[:, :, 1]
    # img_show(img_as)
    img_as = img_as - 5
    combine_img = np.expand_dims(img_as, axis=2)
    combine_img = np.concatenate((combine_img, combine_img, combine_img), axis=-1)
    img_gray = cv2.cvtColor(combine_img, cv2.COLOR_BGR2GRAY)
    # thresh, ret = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # 实测调整为95%效果好一些
    # filter_condition = int(thresh)
    _, a_thresh = cv2.threshold(img_gray, 127, 255, 0)

    contours, hier = cv2.findContours(a_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    area = map(cv2.contourArea, contours)
    area = list(area)
    if len(area) == 0:
        return final_list, contours, post_list

    area_lower = max(area) * 0.8

    for a_index, a in enumerate(area, 0):
        if a > area_lower:
            post_list.append(a_index)

    for post in post_list:
        (x, y), radius = cv2.minEnclosingCircle(contours[post])
        find_flag = 1
        lower_y, upper_y, lower_w, upper_w = 0, 0, 0, 0
        if (y - 0.8 * radius) > 0 and (y + 0.8 * radius) < img_h:
            lower_y = (y - radius) if (y - radius) > 0 else 0
            upper_y = (y + radius) if (y - radius) < img_h else img_h
        else:
            find_flag = 0
        if (x - 0.8 * radius) > 0 and (x + 0.8 * radius) < img_w:
            lower_w = (x - radius) if (x - radius) > 0 else 0
            upper_w = (x + radius) if (x + radius) < img_w else img_w
        if find_flag == 1:
            final = img[int(lower_y):int(upper_y), int(lower_w):int(upper_w)]
            final_list.append(final)

    return final_list, contours, post_list


#  提取印章
def eldsc_extract(out_ellipse: str, img_path, output_path: str):
    ellipses = list()
    img = cv2.imread(img_path)
    no_stamp_flag = 0

    with open(out_ellipse) as out_ellipse:
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

        if ellipse['radians'] > MIN_RADIAN and ellipse['a'] > max_radius:
            print(ellipse, MIN_RADIAN)
            max_radius = ellipse['a']

    temp_count = 0
    arcs_length = list()
    selected_ellipse = []

    for num, ellipse_data in enumerate(ellipses_data, 0):
        if ellipse_data['radians'] > MIN_RADIAN and float(ellipse_data['a']) > (MIN_RADIUS_RATIO * max_radius) \
                and -0.2 < ellipse_data['theta'] < 0.2:
            # 算弧长
            arc_length = ellipse_data['radians'] * ellipse_data['a']
            area = math.pi * ellipse_data['a'] * ellipse_data['b']
            ellipse_data.setdefault('arc_length', arc_length)
            ellipse_data.setdefault('area', area)
            arcs_length.append(arc_length)
            selected_ellipse.append(ellipse_data)
            temp_count += 1

    ring_count = 0
    ring_out = list()
    final_list = list()
    for i, ellipse in enumerate(selected_ellipse, 0):
        flag = 1
        if arcs_length[i] < 0:
            continue

        for j, ellipse_j in enumerate(selected_ellipse[i + 1:], i + 1):
            dx = ellipse['x_c'] - ellipse_j['x_c']
            dy = ellipse['y_c'] - ellipse_j['y_c']
            distance = dx * dx + dy * dy
            para1 = ellipse['b'] if ellipse['b'] < ellipse_j['b'] else \
                ellipse_j['b']
            para1 = para1 * para1
            if distance > para1:
                continue
            else:
                if ellipse['area'] > ellipse_j['area']:
                    arcs_length[j] = -1
                else:
                    flag = 0

        if flag == 1:
            flag_red, _ = select_final_by_red_pixel(ellipse, img)
            if flag_red == 1:
                ring_out.append(ellipse)
                ring_count += 1
    if len(ring_out) != 0:
        max_a = max(ring_out, key=lambda x: x['area'])
        max_a_lower = max_a['area'] * 0.8
        for ring_num, ring in enumerate(ring_out, 0):
            if ring['area'] > max_a_lower:
                _, select_out = select_final_by_red_pixel(ring, img)
                final_out = erase_black_pixel(select_out)
                # img_show(final_out)
                cv2.imwrite(os.path.join(output_path, 'stampELDSC' + str(ring_num) + '.bmp'), final_out)
    else:
        out_list, contours, post_list = LAB_extract(img)
        if len(out_list) != 0:
            for out_index, out in enumerate(out_list, 0):
                final_out = erase_black_pixel(out)
                cv2.imwrite(os.path.join(output_path, 'stampLAB' + str(out_index) + '.bmp'), final_out)
        else:
            no_stamp_flag = 1


def select_final_by_red_pixel(ring, img):
    global red1_range, red2_range
    img_shape = img.shape
    flag = 1
    final = img
    img_w = img_shape[1]
    img_h = img_shape[0]
    red_index = 0
    src_a, src_b = int(ring['a']), int(ring['b'])
    (x, y, w, h) = int(ring['x_c']), int(ring['y_c']), int(ring['a'] * 1.1), int(ring['b'] * 1.1)
    print((x, y, w, h))
    lower_y, upper_y, lower_w, upper_w = 0, 0, 0, 0
    if (y - 0.8 * src_b) > 0 and (y + 0.8 * src_b) < img_h:
        lower_y = (y - h) if (y - h) > 0 else 0
        upper_y = (y + h) if (y - h) < img_h else img_h
    else:
        flag = 0
    if (x - 0.8 * src_a) > 0 and (x + 0.8 * src_a) < img_w:
        lower_w = (x - w) if (x - w) > 0 else 0
        upper_w = (x + w) if (x + w) < img_w else img_w
    else:
        flag = 0
    if flag == 1:
        final = img[lower_y:upper_y, lower_w:upper_w]
        final_shape = final.shape
        final_w = final_shape[1]
        final_h = final_shape[0]
        final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        mask_red_1 = cv2.inRange(final_hsv, red1_range[0], red1_range[1])
        mask_red_2 = cv2.inRange(final_hsv, red2_range[0], red2_range[1])
        mask = mask_red_1 + mask_red_2
        non_zero_pixel = cv2.countNonZero(mask)
        sum_pixel = final_w * final_h
        # img_show(mask)
        red_index = non_zero_pixel / sum_pixel
    print('red index = ' + str(red_index))
    if red_index < 0.02:
        flag = 0
    return flag, final


def resize_img(input_img, size):
    shape = input_img.shape
    w = shape[1]  # 宽度
    h = shape[0]
    if w > h:
        output_img = cv2.resize(input_img, dsize=(int(w * size / h), size))
    else:
        output_img = cv2.resize(input_img, dsize=(size, int(h * size / w)))

    return output_img


def IMG2PGM(img_name, img_path, output_path):
    img = cv2.imread(os.path.join(img_path, img_name), 0)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(img, element1, iterations=1)
    new_name = img_name[:img_name.find('.')] + '.pgm'
    pgm_path = fr"{os.path.join(output_path, new_name)}"
    cv2.imwrite(pgm_path, erosion)
    return pgm_path


if __name__ == '__main__':
    file_list = os.listdir(IMG_INPUT)
    for i, file in enumerate(file_list, 0):
        print(file)
        output_path = make_out_folder(OUT, file)
        # img = cv2.imread(os.path.join(IMG_INPUT, file))
        # img_pgm = cv2.imread(os.path.join(INPUT_DIR, file), 0)
        # dilation = preprocess_(img_pgm)
        # cv2.imwrite('./'+file, dilation)
        pgm_path, bmp_path = preprocess_(file, IMG_INPUT, output_path)
        input_file = os.path
        print(os.system("./elsdc " + pgm_path))
        src_list = [OUT_POLYGON, OUT_ELLIPSE, OUT_SVG]
        copy_out(src_list, output_path)
        # os.remove('./'+file)
        out_ellipse_path = os.path.join(OUT + '/' + file[:file.find('.')], 'out_ellipse.txt')
        eldsc_extract(out_ellipse_path, bmp_path, output_path)
