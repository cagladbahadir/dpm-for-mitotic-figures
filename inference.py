import torch 
import matplotlib.pyplot as plt
import numpy as np
import random
from package_import import *
from dataloader_functions import canine_data_load_test
from denoising_diffusion_pytorch_custom.classifier_free_guidance_real_to_generated import GaussianDiffusion, Unet
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 526
torch.manual_seed(seed) 
random.seed(seed)
np.random.seed(seed)

## Models loaded
diffusion_model = model_select('diffusion', 'hard', 64)
checkpoint = torch.load('model_weights/diffusion_canine/model_checkpoint.pt', map_location = torch.device("cpu"))

diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=[0]).to(device)
diffusion_model.load_state_dict(checkpoint['model_state_dict'])

resnet_models = {}
for i in ['1','2','3']:
    resnet_model = model_select('resnet34', 'hard', 64)
    checkpoint = torch.load('model_weights/resnet/Ensemble_'+i+'/model_checkpoint.pt', map_location = torch.device("cpu"))
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    resnet_models[i] = resnet_model.to(device)

scores = [0,0.5,1]
sigm = torch.nn.Sigmoid()
for score in scores:
    cls = score*torch.ones(1,1).type(torch.float).to(device)
    generated_cell = diffusion_model.module.sample(
                    classes=cls,
                    cond_scale=8.
                    )
    print(generated_cell.shape)
    resnet_scores = torch.zeros(3)
    input_ = torch.nn.functional.interpolate(
            generated_cell.float(),
            scale_factor=(4, 4),
                mode='bilinear')
    input_ = input_.squeeze()
    input_ = input_.unsqueeze(0).float() / torch.quantile(
        input_.unsqueeze(0).float(), 0.99, dim=None)
    with torch.no_grad():
        for i in ['1','2','3']:
            resnet_models[i].eval()
            prediction= sigm(resnet_models[i](input_.to(device)))
            resnet_scores[int(i)-1] = prediction
    if not os.path.isdir('inference_results'): os.makedirs('inference_results')
    diffusion_condition = score
    resnet_score = torch.mean(resnet_scores)
    save_image(generated_cell, 'inference_results/'+'Seed_'+str(seed)+'_Condition_'+str(diffusion_condition)+"_Resnet_score_"+str(np.around(resnet_score.item(),decimals =2))+".png")
                        