# dpm_for_mitotic_figures
Paper: Characterizing the Features of Mitotic Figures Using a Conditional Diffusion Probabilistic Model

Accepted at Deep Generative Models workshop @ MICCAI 2023

## Mitotic Figure Generation
**Aim:**

We train a probabilistic diffusion model to synthesize patches of cell nuclei for a given mitosis label condition. Using this model, we can then generate a sequence of synthetic images that correspond to the same nucleus transitioning into the mitotic state. 

**Generated Cell Examples:**

![Canine_generated_cells](https://github.com/cagladbahadir/dpm-for-mitotic-figures/assets/14334677/9ae01d12-3055-40a5-8212-836ce62bdcdf)

**Figure:** Synthetic cell sets generated with the CCMT DPM. Rows represent cell sets generated from the same random noise, transitioning from non-mitotic (left) to definite mitotic figures (right). Condition values are listed above.

## Data

### Canine Cutaneous Mast Cell Tumor

This dataset comprises of Whole Slide Images and corresponding mitotic and non-mitotic cell annotations.
Dataset can be acquired with permission from: https://github.com/DeepMicroscopy/MITOS_WSI_CCMCT.

**Folder structure:**

WSI images should be stored in: **data/canine/WSI/** (The list of WSIs used for training is listed in data/canine/WSI/CCMT_wsi_list.csv)
**MITOS_WSI_CCMCT_ODAEL.sqlite** should be stored in: **data/canine/Annotations/**

data/canine/ and data/meningioma/ folders both have **slide_reader.py** scripts for reading different type of annotation and WSI data.

### The Digital Brain Tumour Atlas 

This dataset comprises of Whole Slide Images which can be acquired with permission from: https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994.

Corresponding annotations can be requested from cdb232@cornell.edu.

**Folder structure:**

WSI images should be stored in: **data/Meningioma/Brain_Atlas/WSI/** (The list of WSIs used for training is listed in data/canine/WSI/brain_atlas_wsi_list.csv)

Annotations should be stored in: **data/Meningioma/Brain_Atlas/Annotations/**

## Training

### Environment

The conda environment used for the experiments is stored as: **diffusion_for_mitotic_figures.yml**

### Directory Structure

1. **module_loader.py** and **package_import.py** scripts import required packages and set default arguments.
2. **dataloader.py** and **dataloader_functions.py** call data specific loader functions.
3. **trainer.py** script holds the loss and accuracy calculations along with forward pass definition. 
4. **model.py** file calls different types of models.
5. **denoising_diffusion_pytorch_custom** folder is copied from https://github.com/lucidrains/denoising-diffusion-pytorch. Appropriate licensing information is copied in every file under the directory. Minimal change is made in these files.

### Training command

The model can be trained in GPU or CPU, although GPU is highly recommended. The following lines can be used for training the diffusion model or the ResNet models:

**Diffusion model:**

python -u train.py  --date "Current_Date" --optimizer 'Adam' --dataset "canine" --learning_rate 1e-04 --model_type 'diffusion' --batch_size 128 --batch_validation 128 --mite_ratio 0.5 --tile_size 64 --soft_vs_hard 'hard' --ensemble 1

**Resnet model:**

python -u train.py  --date "Current_Date" --optimizer 'Adam' --dataset "canine" --learning_rate 1e-05 --model_type 'resnet34' --loss_function 'BCE' --batch_size 128 --batch_validation 128 --mite_ratio 0.5 --tile_size 64 --soft_vs_hard 'hard' --ensemble 1

## Inference

### Model weights

Model weights for the ResNet model and the diffusion model can be requested from cdb232@cornell.edu.

Diffusion model weights for the canine mitotic figure generation should be stored in the following path: **model_weights/diffusion_canine/model_checkpoint.pt**.

ResNet model weights for classifying generated cell images should be stored in the following path: **model_weights/resnet/Ensemble_n/model_checkpoint.pt** (Three models are ensembled together for the ResNet model)

The generation of a cell from random noise with the diffusion model and their corresponding resnet score can be acquired with the following line. The seed value can be changed to generate different cells.

**Inference:** python -u inference.py 

Inference script generates cell images conditioned on different scores and passes generated cell images from the ResNet classifier. The cell images are saved as pngs for visualization purposes.



