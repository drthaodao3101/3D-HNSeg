import numpy as np
from light_training.dataloading.dataset import get_train_val_loader_from_json
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
from monai.losses import DiceCELoss
set_determinism(123)
import os
import argparse                
import torch.nn.functional as F

#print(torch.is_autocast_enabled())
#print(torch.backends.cuda.flash_sdp_enabled())
#print(torch.backends.cuda.mem_efficient_sdp_enabled())

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class HeneCTTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, roi_size,train_process=8, device="cpu", val_every=1, num_gpus=1,deep_supervision=False, logdir="./logs/",model_save_path="./logs/model/",augmentation=True, master_ip='localhost', master_port=17750, training_script="train.py",checkpoint_path=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script,checkpoint_path=checkpoint_path)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        
        from model_brain_mamba.Brain_Mamba import Brain_Mamba
        self.model = Brain_Mamba(
                in_channels = 1,                                #n_channels
                n_channels = 48,                                #Feature size default for SwinTransformer
                n_classes = 4,                                  #out_classes 
                kernel_sizes=[1,3,5],
                strides=[1,1,1],
                uniform_dec_channels=48,                        #Defaut number of channels in each satge of the decoder
                deep_supervision=deep_supervision,              
                do_res=True,                     
                do_res_up_down = True,
                block_counts = [2,2,2,2,2,2,2,2,2],              #[3,4,4,4,4,4,4,4,3]
                checkpoint_style = None,
                mode='train'                                    #unused
            ).to(self.device)
        
        self.deep_supervision=deep_supervision
        self.augmentation = augmentation
        self.model_save_path=model_save_path
        self.patch_size = roi_size
        self.best_mean_dice = 0.0

        self.loss_fn = DiceCELoss(                                                                              #Loss Define
            to_onehot_y=True,
            softmax=True,
            include_background=False
        ) 
        self.train_process = train_process
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)                                  #Optimzer Define
        self.max_grad_norm = 1.0
        self.scheduler_type = None                                                                            #EfficientMedNeXt dont used Scheduler


    def training_step(self, batch):

        image, label = self.get_input(batch)

        preds = self.model(image)

        loss = 0.0
        if self.deep_supervision==True:
            for pred in preds:
                loss += self.loss_fn(F.interpolate(pred, (label.shape[-3],label.shape[-2],label.shape[-1]), mode='trilinear'), label.unsqueeze(1)) 
        else:
            pred=preds[0]
            loss = self.loss_fn(F.interpolate(pred, (label.shape[-3],label.shape[-2],label.shape[-1]), mode='trilinear'), label.unsqueeze(1)) 
        
        
        self.log("training_loss", loss, step=self.global_step)
        return loss 
    
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
    
        label = label[:, 0].long()
        return image, label

    def on_load_checkpoint(self, extra_state):
        """Restore custom state from checkpoint."""
        if 'best_mean_dice' in extra_state:
            self.best_mean_dice = extra_state['best_mean_dice']
            print(f"Restored best_mean_dice: {self.best_mean_dice}")

    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            #return np.array([1.0, 50])
            return np.array([np.nan, 50])
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
        
        with torch.no_grad():
            logits = self.model(image)              # (B, C, D, H, W)
            logits = F.interpolate(logits[0], (label.shape[-3],label.shape[-2],label.shape[-1]), mode='trilinear')
            pred = logits.argmax(dim=1)              # (B, D, H, W)

        dices = []
        num_classes = 4
        for c in range(1, num_classes):  # remove background
            pred_c = (pred == c).cpu().numpy()
            gt_c   = (label == c).cpu().numpy()

            cal_dice, _ = self.cal_metric(gt_c, pred_c)
            dices.append(cal_dice)

        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs
        abscess, tumor, cyst = np.nanmean(dices[0]), np.nanmean(dices[1]), np.nanmean(dices[2])
        mean_dice = np.nanmean([abscess, tumor, cyst])

        self.log("dice_abscess", abscess, step=self.epoch)
        self.log("dice_tumor", tumor, step=self.epoch)
        self.log("dice_cyst", cyst, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        print(f"Dice: abscess={abscess:.6f}, tumor={tumor:.6f}, cyst={cyst:.6f}")

        if not np.isnan(mean_dice) and mean_dice > self.best_mean_dice:
            print(f"Improve dice from {self.best_mean_dice} to {mean_dice}")
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,                                      #save each epoch that has improvement
                                            os.path.join(self.model_save_path, 
                                            f"best_model_epoch{self.epoch}_{mean_dice:.6f}.pt"), 
                                            delete_symbol="best_model")
            #Save duplication version of current best for inference
            save_new_model_and_delete_last(self.model,                                      
                                            os.path.join(self.model_save_path, 
                                            f"inference_model.pt"), 
                                            delete_symbol="no_need")

            # Save checkpoint when we get a new best model
            checkpoint_path = os.path.join(self.model_save_path, "best_checkpoint.pth")
            self.save_checkpoint(checkpoint_path, best_mean_dice=self.best_mean_dice)

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(self.model_save_path, 
                                        f"final_model_{mean_dice:.6f}.pt"), 
                                        delete_symbol="final_model")

        # Save periodic checkpoint every epoch
        checkpoint_path = os.path.join(self.model_save_path, "latest_checkpoint.pth")
        self.save_checkpoint(checkpoint_path, best_mean_dice=self.best_mean_dice)


        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.6f}.pt"))

        print(f"mean_dice is {mean_dice}")

def parse_args():
    parser = argparse.ArgumentParser(description="Model inference")

    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU id, e.g. 0 or 0,1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fold = args.fold
    num_gpus = args.num_gpus
    gpu = args.gpu

    device = f"cuda:{gpu}"
    max_epoch = 1000 
    out_dir = f"./experiments/henect204/fold_{fold}"
    logdir = os.path.join(out_dir, f"logs") 
    model_save_path = os.path.join(logdir, "model")
    json_path="./data/henect204_seg/splits_final.json"
    data_root = "./data/henect204_seg"
    augmentation = True
    #augmentation = "nomirror"
    env = "pytorch"
    batch_size = 2
    val_every = 2
    train_process=8
    roi_size = [96, 96, 96]
    deep_supervision = False
    trainer = HeneCTTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            roi_size=roi_size,
                            train_process=train_process,
                            device=device,
                            logdir=logdir,
                            model_save_path=model_save_path,
                            augmentation=augmentation,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            deep_supervision=deep_supervision,
                            master_port=17759,
                            training_script=__file__,
                            checkpoint_path=args.checkpoint)

    train_ds, val_ds = get_train_val_loader_from_json(data_root,json_path,fold)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
