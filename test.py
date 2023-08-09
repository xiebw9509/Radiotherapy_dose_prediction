# @paper202305
import os
import shutil
import torch as t
from torch.utils.data import DataLoader
import tqdm
from torchnet import meter
from PIL import Image, ImageDraw
from skimage import exposure, measure
import numpy as np
import colorsys

from models.Unet3D import UNet3D
from data.dataset import TestDataset
from configs import opt
from utils.gradient_color_generater import SimpleGradientColorGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = opt.test_gpu_index

min_v = -1000
max_v = 3000

dose_width = opt.max_dose - opt.min_dose

colors = ["#0000FF", "#00FFFF", "#90EE90", "#FFFF00", "#FFA500", "#FF0000"]
# rois_name = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
#              'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63', 'PTV70']
#
# metrics_name = ['D95', 'D98', 'D99', 'Dmean', 'Dmax']
# modes_name = ['GT', 'Pred', 'Diff']

def test():
    os.makedirs(opt.predict_result_dir, exist_ok=True)
    color_generator = SimpleGradientColorGenerator(colors, interval_count=30, is_rgb=False)
    color_generator.save_to_image(os.path.join(opt.predict_result_dir, 'colorbar.png'),
                                  [10, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], opt.min_dose, opt.max_dose)

    # model
    model = UNet3D(1)
    # cuda
    model = model.cuda()
    # test
    model.eval()

    # load data
    test_data = TestDataset(opt.data_root, opt.resample_data_folder, 'test_series.txt')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # meter
    diff_meter = meter.AverageValueMeter()

    print("Inferring epoch_{}".format(opt.test_epoch_id))

    # load params
    checkpoint_path = os.path.join(opt.checkpoints_dir, "checkpoint_{}.pth".format(opt.test_epoch_id))
    checkpoint_params = t.load(checkpoint_path, map_location=lambda storage, loc: storage.cpu())
    model.load(checkpoint_params['state_dict'])

    # color
    palette = get_palette()
    # save image
    win_width = opt.win_width
    win_center = opt.win_center

    with t.no_grad():
        for index, (image_data, label_data, dose_data, distance_map, all_masks, origin_image, series_uid) in tqdm.tqdm(enumerate(test_loader)):
            print(origin_image.shape)
            # NxCxDxHxW, origal image
            img_size = image_data.shape

            image_data = image_data.cuda()
            label_data = label_data.cuda()
            dose_data = dose_data.cuda()
            distance_map = distance_map.cuda()

            pred_dose = model.predict(image_data, label_data, distance_map)

            diff = t.sum(opt.max_dose * t.abs(dose_data -  pred_dose)) / (image_data.shape[2] * image_data.shape[3] * image_data.shape[4])
            diff_meter.add(diff.data.cpu().numpy())

            # save results path
            series_uid_dir = os.path.join(opt.predict_result_dir, series_uid[0])
            os.makedirs(series_uid_dir, exist_ok=True)

            # generate vis
            if opt.generate_png:

                image_data = origin_image.numpy()[0, 0, ...]
                dose_data = dose_data.cpu().numpy()[0, 0, ...] * opt.max_dose
                pred_dose_abs = pred_dose.cpu().numpy()[0, 0, ...] * opt.max_dose
                all_masks = all_masks.numpy()[0,...]

                # save vis path
                png_dir = os.path.join(opt.predict_result_dir, series_uid[0], 'image')

                create_dose_images(image_data, dose_data, pred_dose_abs, all_masks,
                                   png_dir, opt.min_dose, dose_width, win_center, win_width, palette, color_generator)

            if opt.save_volume:
                import SimpleITK as sitk
                pred_dose_abs = pred_dose.cpu().numpy()[0, 0, ...] * opt.max_dose
                itk_dose = sitk.GetImageFromArray(pred_dose_abs)
                itk_dose.SetSpacing((2.5, 2.5, 3))
                volume_dir = os.path.join(opt.predict_result_dir, series_uid[0], 'volume')
                os.makedirs(volume_dir, exist_ok=True)
                sitk.WriteImage(itk_dose, os.path.join(volume_dir, 'dose.nii.gz'))


        save_diff_for_all(diff_meter, os.path.join(opt.predict_result_dir, 'eval_diff.txt'))

        # save scripts
        copy_scripts('.', opt.predict_dir)
        # copy best_snapshot
        best_snapshot_name = os.path.join(opt.predict_dir, "checkpoint_{}-publish.pth".format(opt.test_epoch_id))
        t.save(checkpoint_params['state_dict'], best_snapshot_name)

def save_diff_for_all(diff_meter, save_path):
    with open(save_path, 'w') as f:
        print('diff for all:', file=f)
        print('ave:{} std:{}'.format(diff_meter.mean, diff_meter.std), file=f)

def save_image(image, path):
    dir = path[0: path.rfind('/') + 1]
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save(path)

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

    for label_index in range(opt.num_oars + opt.num_targets + 1):
        label = labels[label_index]

        contours = measure.find_contours(label, 0.5)
        color = palette[label_index]

        for contour in contours:
            trans_contour = []
            for index, item in enumerate(contour):
                trans_contour.append((contour[index][1], contour[index][0]))
                draw.line(tuple(trans_contour), fill=tuple(color), width=1)

    del draw
    return output

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

def copy_scripts(src_dir, tar_dir):
    '''
    Copy files from the source directory to the destination directory to save all training-related scripts
    :param src_dir: source path
    :param tar_dir: target path
    '''
    for object in os.listdir(src_dir):
        if object != 'train' and object != '.idea':
            copy_object(object, src_dir, tar_dir)

def copy_object(object_name, src_dir, tar_dir):
    '''
    Copy a file or directory from the source directory to the target directory
    :param object_name: files name， path
    :param src_dir: source path
    :param tar_dir: target path
    '''
    src_path = os.path.join(src_dir, object_name)
    tar_path = os.path.join(tar_dir, object_name)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, tar_path)
    else:
        shutil.copyfile(src_path, tar_path)

def get_palette():
    '''

    :return: (numpy array) 
    '''

    # h_steps x sv_steps （number of color）
    h_steps = 6;
    sv_steps = 6;

    palette = []
    # the background setting negative
    palette.append((-1, -1, -1))

    for v in range(sv_steps, 0, -1):
        for s in range(sv_steps, 0, -1):
            for h in range(0, h_steps):
                hsv_value = (h / h_steps, s / sv_steps, v / sv_steps)
                rgb_value = colorsys.hsv_to_rgb(*(hsv_value))
                palette.append((int(rgb_value[0] * 255), int(rgb_value[1] * 255), int(rgb_value[2] * 255)))

    palette = np.array(palette)
    return  palette

def create_dose_images(ct, dose, pred_dose, rois_mask, save_dir, min_dose, dose_width, win_center, win_width, palette,
                       color_generator):
    # transpose #  c,z,y,x

    # 归一化
    dose = (dose - min_dose) / dose_width
    pred_dose = (pred_dose - min_dose) / dose_width
    dose_rgb = create_dose_rgb(dose, color_generator)
    pred_dose_rgb = create_dose_rgb(pred_dose, color_generator)
    # tranverse
    transverse_dir = os.path.join(save_dir, 'transverse')
    os.makedirs(transverse_dir, exist_ok=True)
    start = int(0 * ct.shape[0])
    end = int(0.9 * ct.shape[0])
    for z in range(start, end):
        ct_slice = ct[z, :, :]
        rois_mask_slice = rois_mask[:, z, :, :]
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
        ct_slice = ct[::-1, :, x]
        rois_mask_slice = rois_mask[:, ::-1, :, x]
        dose_slice = dose_rgb[::-1, :, x]
        pred_dose_slice = pred_dose_rgb[::-1, :, x]

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
        ct_slice = ct[::-1, y, :]
        rois_mask_slice = rois_mask[:, ::-1, y, :]
        dose_slice = dose_rgb[::-1, y, :]
        pred_dose_slice = pred_dose_rgb[::-1, y, :]

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


if __name__ == '__main__':
    test()


