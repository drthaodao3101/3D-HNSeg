from light_training.dataloading.dataset import get_train_val_loader_from_json_for_cross_validation
from monai.utils import set_determinism
import torch 
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm 
import json

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
        class_wise_metric[cls, ...] = cal_metric((pred==(cls+1)), (gt==(cls+1)), voxel_spacing)     #cls+1 => remove background
    # Tạm tắt print ở đây để terminal đỡ bị trôi khi chạy nhiều fold
    # print(class_wise_metric, flush=True)
    return class_wise_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính toán metric (Dice, IoU, HD95) cho các fold cụ thể.")
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4], 
                        help='Danh sách các fold muốn chạy (vd: 0 1 2). Mặc định là 0 1 2 3 4.')
    
    args = parser.parse_args()
    folds_to_run = args.folds

    data_root = "./data/henect204_seg"
    raw_data_dir = "./data/henect204_seg/gt_segmentations"
    
    for fold in folds_to_run:
        print(f"\n{'='*20} BẮT ĐẦU TÍNH METRIC CHO FOLD {fold} {'='*20}")
        
        # results_root = f"./experiments/henect204/prediction_results_fold{fold}" 
        results_root = f"./experiments/henect204/prediction_results" 
        json_path = f"./data/henect204_seg/splits_fold{fold}.json"  
        
        if not os.path.exists(json_path):
            print(f"[Cảnh báo] Không tìm thấy {json_path}. Bỏ qua fold {fold}.")
            continue


        with open(json_path, 'r') as f:
            js_data = json.load(f)
            
        # Tùy thuộc file json của bạn là list (chứa 1 dict) hay trực tiếp là dict
        if isinstance(js_data, list):
            val_cases = js_data[0]["val"] 
        else:
            val_cases = js_data["val"]

        num_cases = len(val_cases)
        print(f"Số lượng case trong fold {fold}: {num_cases}")
        
        all_results = np.zeros((num_cases, 3, 3)) 
        ind = 0
        
        # Duyệt thẳng qua danh sách tên case thay vì qua Dataloader
        for case in tqdm(val_cases, total=num_cases, desc=f"Processing Fold {fold}"):
            # Xử lý format tên case nếu trong JSON nó lưu dưới dạng dict thay vì string
            case_name = case if isinstance(case, str) else case["name"]
            
            gt_itk_path = os.path.join(raw_data_dir, f"{case_name}.nrrd")
            gt_itk = sitk.ReadImage(gt_itk_path)
            
            # Lấy spacing trực tiếp từ ITK image (chính xác hơn)
            voxel_spacing = gt_itk.GetSpacing()
            gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
            
            pred_itk_path = f"./{results_root}/{case_name}.nrrd"
            pred_itk = sitk.ReadImage(pred_itk_path)
            pred_array = sitk.GetArrayFromImage(pred_itk)
            
            m = each_cases_metric(gt_array, pred_array, voxel_spacing)
            all_results[ind, ...] = m
            ind += 1

        # Lưu và in kết quả
        save_dir = f"./experiments/henect204/result_metrics_fold{fold}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "all_results.npy")
        np.save(save_path, all_results) 
        
        result = np.load(save_path)
        cls_mean = np.nanmean(result, axis=0)
        
        print(f"\n--- TỔNG KẾT FOLD {fold} ---")
        print("- Mean Dice, IoU, HD95 for EACH class:")
        print(cls_mean)
        print("- Mean Dice, IoU, HD95 for ALL classes:")
        print(cls_mean.mean(axis=0))
        print("="*65 + "\n")