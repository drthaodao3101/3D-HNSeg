import numpy as np
from light_training.dataloading.dataset import get_train_val_loader_from_json_for_validation
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
set_determinism(123)
import os
from light_training.prediction import Predictor
import argparse

class HeneCTEvaluater(Trainer):
    def __init__(self, 
                 env_type, 
                 max_epochs, 
                 batch_size,
                 patch_size,
                 device="cpu", 
                 val_every=1, 
                 num_gpus=1, 
                 logdir="./logs/", 
                 master_ip='localhost', 
                 master_port=17750,
                 model_checkpoint_path=None,
                 save_path="./prediction_results", 
                 training_script="train.py"):
        
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.save_path=save_path
        self.model_checkpoint_path=model_checkpoint_path
        self.patch_size = patch_size
        self.augmentation = False
    

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        #label = self.convert_labels(label)
        label=label.long()
        return image, label, properties 

    def define_model_segmamba(self):
        from model_brain_mamba.Brain_Mamba import Brain_Mamba
        model = Brain_Mamba(
                in_channels = 1,                                #n_channels
                n_channels = 48,                                #Feature size default for SwinTransformer
                n_classes = 4,                                  #out_classes 
                kernel_sizes=[1,3,5],
                strides=[1,1,1],
                uniform_dec_channels=48,                        #Defaut number of channels in each satge of the decoder
                deep_supervision=False,              
                do_res=True,                     
                do_res_up_down = True,
                block_counts = [2,2,2,2,2,2,2,2,2],              #[3,4,4,4,4,4,4,4,3]
                checkpoint_style = None,
                mode='train'                                    #unused
            ).to(self.device)
        
        model_checkpoint_path = self.model_checkpoint_path
        new_sd = self.filte_state_dict(torch.load(model_checkpoint_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        os.makedirs(self.save_path, exist_ok=True)

        return model, predictor, save_path
    
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        ddim = False
      
        model, predictor, save_path = self.define_model_segmamba()

        model_output = predictor.maybe_mirror_and_predict(image, model, device=self.device)

        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties)
        
        model_output = model_output.argmax(dim=0)#[None]

        label = label[0]
        c = 4
        dices = []
        for i in range(1, c):
            output_i = (model_output==i).cpu().numpy()
            label_i = (label==i).cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)

        print(dices)

        model_output = predictor.predict_noncrop_probability(model_output, properties)

        
        predictor.save_to_nrrd(model_output, 
                              raw_spacing=properties['sitk_stuff']['spacing'],
                              raw_origin=properties['sitk_stuff']['origin'],
                              raw_direction=properties['sitk_stuff']['direction'],
                              case_name = properties['name'][0],
                              save_dir=save_path)
        return 0

    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd

def parse_args():
    parser = argparse.ArgumentParser(description="Model inference")

    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index")

    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs")

    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU id, e.g. 0 or 0,1")

    parser.add_argument("--model_checkpoint_path", type=str, default="",
                        help="Path to trained model checkpoint, if path is not specific, inference_model.pt is used")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    fold = args.fold
    num_gpus = args.num_gpus
    gpu = args.gpu
    model_checkpoint_path = args.model_checkpoint_path
    if model_checkpoint_path == "":
        model_checkpoint_path = f"./experiments/henect204/fold_{fold}/logs/model/inference_model.pt"
    
    data_root = "./data/henect204_seg"
    json_path="./data/henect204_seg/splits_final.json"

    save_path = f"./experiments/henect204/prediction_results"

    env = "pytorch"
    max_epoch = 1000
    batch_size = 2
    val_every = 2
    device = f"cuda:{gpu}"
    patch_size = [96, 96, 96]

    trainer = HeneCTEvaluater(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            patch_size = patch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            model_checkpoint_path = model_checkpoint_path,
                            save_path=save_path,
                            training_script=__file__)
    

    val_ds = get_train_val_loader_from_json_for_validation(data_root,json_path,fold)
    trainer.validation_single_gpu(val_ds)


