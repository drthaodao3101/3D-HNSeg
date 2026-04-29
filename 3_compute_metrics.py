from light_training.dataloading.dataset import get_train_val_loader_from_json_for_cross_validation
from monai.utils import set_determinism
import torch 
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm 

import numpy as np

set_determinism(123)

def cal_metric(gt, pred, voxel_spacing):
    gt_sum = gt.sum()
    pred_sum = pred.sum()

    # both empty → undefined
    if gt_sum == 0 and pred_sum == 0:
        return np.array([np.nan, np.nan, np.nan])

    # one empty, one non-empty
    if gt_sum == 0 or pred_sum == 0:
        return np.array([0.0, 0.0, np.nan])

    # both non-empty
    dice = metric.binary.dc(pred, gt)
    iou  = metric.binary.jc(pred, gt)   # IoU = Jaccard
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
    return np.array([dice, iou, hd95])

def each_cases_metric(gt, pred, voxel_spacing):
    classes_num = 3             # remove background
    class_wise_metric = np.zeros((classes_num, 3))                  #dice, iou, hd95
    for cls in range(0, classes_num):
        #class_wise_metric[cls, ...] = cal_metric(pred[cls], gt[cls], voxel_spacing)
        class_wise_metric[cls, ...] = cal_metric((pred==(cls+1)), (gt==(cls+1)), voxel_spacing)     #cls+1 => remove background
    print(class_wise_metric, flush=True)
    return class_wise_metric


if __name__ == "__main__":

    results_root = "./experiments/henect204/prediction_results"
    #data_dir = "./data/fullres/train"
    data_root = "./data/henect204_seg"
    json_path="./data/henect204_seg/splits_final.json"  
    #raw_data_dir = "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    raw_data_dir = "./data/henect204_seg/gt_segmentations"
    #train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    test_ds = get_train_val_loader_from_json_for_cross_validation(data_root,json_path)
    print(len(test_ds))
    all_results = np.zeros((204,3,3))                           # 204 case, 3 class, and [Dice, IoU, Hausdorff95]
    ind = 0
    for batch in tqdm(test_ds, total=len(test_ds)):
        properties = batch["properties"]
        case_name = properties["name"]
        gt_itk = os.path.join(raw_data_dir,f"{case_name}.nrrd")
        voxel_spacing = properties['sitk_stuff']['spacing']
        gt_itk = sitk.ReadImage(gt_itk)
        voxel_spacing = gt_itk.GetSpacing()
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        pred_itk = sitk.ReadImage(f"./{results_root}/{case_name}.nrrd")
        voxel_spacing = gt_itk.GetSpacing()
        pred_array = sitk.GetArrayFromImage(pred_itk)
        m = each_cases_metric(gt_array, pred_array, voxel_spacing)

        all_results[ind, ...] = m
        ind += 1

    os.makedirs(f"./experiments/henect204/result_metrics/", exist_ok=True)
    np.save(f"./experiments/henect204/result_metrics/all_results.npy", all_results) 
    
    result = np.load(f"./experiments/henect204/result_metrics/all_results.npy")
    print(result.shape)
    cls=np.nanmean(result, axis=0)
    print("- Dice, IoU, HD95 for each class:")
    print(cls)
    #print("- Std Dice, IoU, HD95 for each class:")
    #print(np.nanstd(result, axis=0))
    print("- Mean Dice, IoU, HD95 for all classes:")
    print(cls.mean(axis=0))


