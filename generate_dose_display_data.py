#-*-coding:utf-8 -*-
# @paper202305
import os
import numpy as np
from PIL import Image, ImageDraw
from skimage import exposure, measure
import tqdm
import pandas as pd

from configs import opt
from utils.gradient_color_generater import SimpleGradientColorGenerator
from provided_code.data_loader import DataLoader
from provided_code.general_functions import get_paths

import matplotlib.pyplot as plt

def generate():
    win_width = 350
    win_center = 80
    min_v = -1000
    max_v = 3000
    max_dose = 8000
    min_dose = 10
    dose_width = max_dose - min_dose

    colors = ["#0000FF", "#00FFFF",  "#90EE90", "#FFFF00", "#FFA500", "#FF0000"]
    rois_name = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                 'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63', 'PTV70']

    metrics_name = ['D95', 'D98', 'D99', 'Dmean', 'Dmax']
    modes_name = ['GT', 'Pred', 'Diff']

    # dir setting
   
    predict_dir = '/Data/CZJ/DoseDistributionPrediction/OpenKBP_Master/train/train1-resunet-bn-target-fill-distance_map-mean-batch4/predict/predict_1'

    if not os.path.exists(predict_dir):
        raise Exception("predict_dir not exists");

    prediction_res_dir = os.path.join(predict_dir, 'result')

    if not os.path.exists(prediction_res_dir):
        raise Exception("prediction_res_dir not exists");

    prediction_res_image_dir = os.path.join(predict_dir, 'result_imgs')
    os.makedirs(prediction_res_image_dir, exist_ok=True)

    palette = get_palette()
    color_generator = SimpleGradientColorGenerator(colors, interval_count=30, is_rgb=False)
    color_generator.save_to_image(os.path.join(prediction_res_image_dir, 'colorbar.png'), [0.1, 10, 20, 30, 40, 50, 60, 70, 80], min_dose, max_dose)

    training_data_dir = '{}/train-pats'.format(opt.primary_directory)
    plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
    num_train_pats = np.minimum(opt.num_train_pats, len(plan_paths))  # number of plans that will be used to train model

    val_paths = plan_paths[num_train_pats:]

    data_loader_hold_out_eval = DataLoader(val_paths, batch_size=1, mode_name='training_model')  # Set data loader
    prediction_paths = get_paths(prediction_res_dir, ext='csv')
    hold_out_prediction_loader = DataLoader(prediction_paths, batch_size=1, mode_name='predicted_dose')  # Set prediction loader

    num_batches = data_loader_hold_out_eval.number_of_batches()

    rois_stat_gt = {}
    rois_stat_pred = {}
    rois_stat_diff = {}

    init_rois_stat(rois_name, rois_stat_gt)
    init_rois_stat(rois_name, rois_stat_pred)
    init_rois_stat(rois_name, rois_stat_diff)

    for idx in tqdm.tqdm(range(num_batches)):
        idx = idx
        rois_batch = data_loader_hold_out_eval.get_batch(idx)
        ct = np.squeeze(rois_batch['ct'] + min_v)
        rois_mask = rois_batch['structure_masks'][0]
        rois_mask[:, :, :, 8] = rois_mask[:, :, :, 8] + rois_mask[:, :, :, 9]
        rois_mask[:, :, :, 7] = rois_mask[:, :, :, 7] + rois_mask[:, :, :, 8] + rois_mask[:, :, :, 9]

        patient_list = rois_batch['patient_list']

        dose = np.squeeze(rois_batch['dose'])
        rois_batch_predict = hold_out_prediction_loader.get_batch(patient_list=patient_list)
        pred_dose = np.squeeze(rois_batch_predict['predicted_dose'])

        # create patient dir
        save_dir = os.path.join(prediction_res_image_dir, patient_list[0])

        # create dose imags
        create_dose_images(ct, dose, pred_dose, rois_mask, save_dir, min_dose, dose_width, win_center, win_width,
                           palette, color_generator)

        # create dvh and calulate some metric_eval
        calc_dvh_and_metrics(dose, pred_dose, rois_mask, rois_name, rois_stat_gt, rois_stat_pred, rois_stat_diff,
                             max_dose, save_dir)

    do_statistics(rois_stat_gt)
    do_statistics(rois_stat_pred)
    do_statistics(rois_stat_diff)

    create_table(metrics_name, modes_name, prediction_res_image_dir, rois_name, rois_stat_gt, rois_stat_pred,
                 rois_stat_diff)

def create_table(metrics_name, modes_name, save_dir, rois_name, rois_stat_gt, rois_stat_pred, rois_stat_diff):
    fusion_names = []
    for metric_name in metrics_name:
        for mode_name in modes_name:
            fusion_names.append('{}_{}'.format(metric_name, mode_name))
    rois_stat = {}
    rois_stat['GT'] = rois_stat_gt
    rois_stat['Pred'] = rois_stat_pred
    rois_stat['Diff'] = rois_stat_diff
    data = {}
    for metric_name in metrics_name:
        for mode_name in modes_name:
            dir_mode = []
            temp_rois_stat = rois_stat[mode_name]

            for roi_name in rois_name:
                mean, std = temp_rois_stat[roi_name][metric_name]
                #dir_mode.append("{}({})".format(("%.3f" % mean), ("%.3f" % std)))
                dir_mode.append("{}".format("%.3f" % mean))

            data['{}_{}'.format(metric_name, mode_name)] = dir_mode
    df = pd.DataFrame(data, index=rois_name, columns=fusion_names)
    df.to_csv(os.path.join(save_dir, 'table.csv'))


def calc_dvh_and_metrics(dose, pred_dose, rois_mask, rois_name, rois_stat_gt, rois_stat_pred, rois_stat_diff,
                         max_dose, save_dir):
    save_dvh_dir = os.path.join(save_dir, 'dvh')
    os.makedirs(save_dvh_dir, exist_ok=True)
    for i in range(rois_mask.shape[3]):
        mask = rois_mask[:, :, :, i]
        if np.max(mask) < 1:
            continue
        dose_mask = dose[mask.astype(bool)]
        pred_dose_mask = pred_dose[mask.astype(bool)]

        # 归一化
        dose_mask = dose_mask
        pred_dose_mask = pred_dose_mask

        dvh_volume = []
        dvh_dose = []
        dvh_volume_pred = []
        dvh_dose_pred = []

        for V_percent in range(0, 100 + 1):
            dvh_volume.append(V_percent)
            dvh_volume_pred.append(V_percent)

            D = np.percentile(dose_mask, 100 - V_percent)
            dvh_dose.append(D)
            D_pred = np.percentile(pred_dose_mask, 100 - V_percent)
            dvh_dose_pred.append(D_pred)

        dvh_dose = np.array(dvh_dose)
        dvh_dose_pred = np.array(dvh_dose_pred)
        plt.figure()
        plt.plot(dvh_dose, dvh_volume, label='ground truth')
        plt.plot(dvh_dose_pred, dvh_volume_pred, label='predition')
        plt.legend()
        plt.xlim([0, max_dose])
        plt.ylim([0, 100])
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')
        plt.title(rois_name[i])

        plt.savefig('{}/{}.png'.format(save_dvh_dir, rois_name[i]))
        plt.close()

        eval_metric(rois_stat_gt[rois_name[i]], dose_mask, dvh_dose)
        eval_metric(rois_stat_pred[rois_name[i]], pred_dose_mask, dvh_dose_pred)
        eval_metric_diff(rois_stat_diff[rois_name[i]], rois_stat_gt[rois_name[i]], rois_stat_pred[rois_name[i]])


def do_statistics(rois_stat):
    for roi_name in rois_stat.keys():
        roi_stat = rois_stat[roi_name]

        for field_name in roi_stat.keys():
            mean, std = cal_mean_and_std(roi_stat[field_name])
            roi_stat[field_name] = (mean, std)

def cal_mean_and_std(value_list):
    arr = np.array(value_list)
    std = np.std(arr)
    mean = np.mean(arr)

    return mean, std

def eval_metric_diff(roi_stat_diff, roi_stat, roi_stat_pred):
    mean = abs(roi_stat['Dmean'][-1] - roi_stat_pred['Dmean'][-1])
    max = abs(roi_stat['Dmax'][-1] - roi_stat_pred['Dmax'][-1])
    D_99 = abs(roi_stat['D99'][-1] - roi_stat_pred['D99'][-1])
    D_98 = abs(roi_stat['D98'][-1] - roi_stat_pred['D98'][-1])
    D_95 = abs(roi_stat['D95'][-1] - roi_stat_pred['D95'][-1])

    roi_stat_diff['Dmean'].append(mean)
    roi_stat_diff['Dmax'].append(max)
    roi_stat_diff['D99'].append(D_99)
    roi_stat_diff['D98'].append(D_98)
    roi_stat_diff['D95'].append(D_95)

def eval_metric(roi_stat, dose_mask, dvh_dose):
    mean = dose_mask.mean()
    max = dose_mask.max()
    D_99 = dvh_dose[99]
    D_98 = dvh_dose[98]
    D_95 = dvh_dose[95]

    roi_stat['Dmean'].append(mean)
    roi_stat['Dmax'].append(max)
    roi_stat['D99'].append(D_99)
    roi_stat['D98'].append(D_98)
    roi_stat['D95'].append(D_95)

def init_rois_stat(rois_name, rois_stat):
    for roi_name in rois_name:
        roi_stat = {}
        roi_stat['Dmean'] = []
        roi_stat['Dmax'] = []
        roi_stat['D95'] = []
        roi_stat['D98'] = []
        roi_stat['D99'] = []

        rois_stat[roi_name] = roi_stat


def create_dose_images(ct, dose, pred_dose, rois_mask, save_dir, min_dose, dose_width, win_center, win_width, palette,
                       color_generator):
    # transpose #  y,x,z,c -> z,y,x,c
    dose = dose.transpose(2, 0, 1)
    pred_dose = pred_dose.transpose(2, 0, 1)
    ct = ct.transpose(2, 0, 1)
    rois_mask = rois_mask.transpose(2, 0, 1, 3)
    # 归一化
    dose = (dose - min_dose) / dose_width
    pred_dose = (pred_dose - min_dose) / dose_width
    dose_rgb = create_dose_rgb(dose, color_generator)
    pred_dose_rgb = create_dose_rgb(pred_dose, color_generator)
    # tranverse
    transverse_dir = os.path.join(save_dir, 'transverse')
    os.makedirs(transverse_dir, exist_ok=True)
    start = int(0.25 * ct.shape[0])
    end = int(0.75 * ct.shape[0])
    for z in range(start, end):
        ct_slice = ct[z, :, :]
        rois_mask_slice = rois_mask[z, :, :, :]
        dose_slice = dose_rgb[z, :, :]
        pred_dose_slice = pred_dose_rgb[z, :, :]
        # y,x,z,c
        bmp_ct = convert_med_img(ct_slice, win_width, win_center)
        merge_contour = merge_with_ct_contours(bmp_ct, rois_mask_slice.astype(np.int32), palette)
        merge_dose = Image.blend(Image.fromarray(dose_slice.astype(np.uint8)),
                                 Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)
        merge_pred_dose = Image.blend(Image.fromarray(pred_dose_slice.astype(np.uint8)),
                                      Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)

        res_img = splice_images(merge_contour, merge_dose, merge_pred_dose)
        res_img.save(os.path.join(transverse_dir, "{}.png".format(z)))

    # sagittal
    sagittal_dir = os.path.join(save_dir, 'sagittal')
    os.makedirs(sagittal_dir, exist_ok=True)
    start = int(0.25 * ct.shape[2])
    end = int(0.75 * ct.shape[2])

    for x in range(start, end):
        ct_slice = ct[:, :, x]
        rois_mask_slice = rois_mask[:, :, x, :]
        dose_slice = dose_rgb[:, :, x]
        pred_dose_slice = pred_dose_rgb[:, :, x]

        # y,x,z,c
        bmp_ct = convert_med_img(ct_slice, win_width, win_center)
        merge_contour = merge_with_ct_contours(bmp_ct, rois_mask_slice.astype(np.int32), palette)
        merge_dose = Image.blend(Image.fromarray(dose_slice.astype(np.uint8)),
                                 Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)
        merge_pred_dose = Image.blend(Image.fromarray(pred_dose_slice.astype(np.uint8)),
                                      Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)

        res_img = splice_images(merge_contour, merge_dose, merge_pred_dose)
        res_img.save(os.path.join(sagittal_dir, "{}.png".format(x)))

    # coronal
    coronal_dir = os.path.join(save_dir, 'coronal')
    os.makedirs(coronal_dir, exist_ok=True)
    start = int(0.25 * ct.shape[1])
    end = int(0.75 * ct.shape[1])

    for y in range(start, end):
        ct_slice = ct[:, y, :]
        rois_mask_slice = rois_mask[:, y, :, :]
        dose_slice = dose_rgb[:, y, :]
        pred_dose_slice = pred_dose_rgb[:, y, :]

        # y,x,z,c
        bmp_ct = convert_med_img(ct_slice, win_width, win_center)
        merge_contour = merge_with_ct_contours(bmp_ct, rois_mask_slice.astype(np.int32), palette)
        merge_dose = Image.blend(Image.fromarray(dose_slice.astype(np.uint8)),
                                 Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)
        merge_pred_dose = Image.blend(Image.fromarray(pred_dose_slice.astype(np.uint8)),
                                      Image.fromarray(bmp_ct.astype(np.uint8)), 0.4)

        res_img = splice_images(merge_contour, merge_dose, merge_pred_dose)
        res_img.save(os.path.join(coronal_dir, "{}.png".format(y)))


def create_dose_rgb(dose, color_generator):
    dose_rgb = np.zeros((*dose.shape, 3))

    for i in range(dose.shape[0]):
        for j in range(dose.shape[1]):
            for k in range(dose.shape[2]):
                dose_value = dose[i, j, k]

                if dose_value >= 0:
                    color = color_generator.get_rgb_color(dose_value)

                    dose_rgb[i, j, k, 0] = color[0]
                    dose_rgb[i, j, k, 1] = color[1]
                    dose_rgb[i, j, k, 2] = color[2]

    return dose_rgb.astype(np.uint8)


def merge(ct_data, labels,
          win_center, win_width, palette):

    bmp_ct = convert_med_img(ct_data, win_width, win_center)

    merge_predict = merge_with_ct_contours(bmp_ct, labels, palette)

    return  merge_predict

def convert_med_img(in_data, win_width, win_center):

    in_re = exposure.rescale_intensity(in_data, in_range=(win_center - win_width / 2.0, win_center + win_width / 2.0),
                                       out_range=(0, 255))
    in_re = np.array(in_re, dtype=np.uint8)
    in_rgb = in_re[:, :, np.newaxis].repeat(3, axis=2)
    return  in_rgb

def merge_with_ct_contours(bmp_ct, labels, palette):
    output = Image.fromarray(bmp_ct.copy())
    draw = ImageDraw.Draw(output)

    for label_index in range(labels.shape[2]):
        label = labels[:, :, label_index]
        contours = measure.find_contours(label, 0.5)
        color = palette[label_index]

        for contour in contours:
            trans_contour = []
            for index, item in enumerate(contour):
                trans_contour.append((contour[index][1], contour[index][0]))
                draw.line(tuple(trans_contour), fill=tuple(color), width=1)

    del draw
    return output

def get_palette():
    palette = []
    # ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
    #  'Esophagus', 'Larynx', 'Mandible'], targets = ['PTV56', 'PTV63', 'PTV70'])
    palette.append((95, 160, 6))
    palette.append((149, 91, 250))
    palette.append((247, 134, 96))
    palette.append((158, 162, 82))
    palette.append((250, 163, 100))
    palette.append((226, 76, 112))
    palette.append((231, 244, 16))

    palette.append((0, 255, 0))
    palette.append((255, 255, 0))
    palette.append((255, 0, 0))

    palette = np.array(palette)
    return  palette

def extract_contours(input, class_index):
    input_c = input.copy()

    input_c[input_c != class_index] = 0
    contours = measure.find_contours(input_c, class_index-0.5)
    return contours

def splice_images(image1, image2, image3):
    width, height = image1.size
    out = Image.new(image1.mode, ((width + 20)*3 , height), color=(255, 255, 255))

    start_width = 10
    out.paste(image1, box=(start_width, 0))
    start_width += width + 20
    out.paste(image2, box=(start_width, 0))
    start_width += width + 20
    out.paste(image3, box=(start_width, 0))

    return out

if __name__ == '__main__':
    generate()







