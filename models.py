import numpy as np
import torch
from torchvision.models import resnet34
import sys
from denoising_diffusion_pytorch_custom.classifier_free_guidance import Unet, GaussianDiffusion

def model_select(model_type, soft_vs_hard, image_dim):

    if 'resnet34' in model_type:

        model = resnet34(pretrained=True)
        model.fc = torch.nn.Linear(in_features = 512, out_features = 1, bias = True)

    elif model_type == 'diffusion':

        if soft_vs_hard == 'hard':
            num_classes = 2
        else:
            num_classes = 3

        model = Unet(
            dim=image_dim,
            dim_mults=(1, 2, 4, 8),
            num_classes=num_classes,
            cond_drop_prob=0.5
        )

        model = GaussianDiffusion(
            model,
            image_size=64,
            timesteps=1000,  # number of steps
            loss_type='l1'  # L1 or L2
        )
    return model

def checkpoint_history(**args):


    hist = np.load(args['save_dir_model']+"/History.npz")
    train_loss = np.append(hist['train_loss'],args['train_loss']/args['len_training_ids'] )
    validation_loss = np.append(hist['validation_loss'], args['validation_loss']/args['len_validation_ids'])
    train_accuracy = np.append(hist['train_accuracy'],args['train_accuracy'].cpu()/args['len_training_ids'])
    validation_accuracy = np.append(hist['validation_accuracy'], args['validation_accuracy'].cpu() / args['len_validation_ids'])

    train_accuracy_true_pos = np.append(hist['train_accuracy_true_pos'],args['train_accuracy_true_pos'].cpu()/args['len_training_ids'])
    train_accuracy_true_neg = np.append(hist['train_accuracy_true_neg'],args['train_accuracy_true_neg'].cpu()/args['len_training_ids'])
    train_accuracy_false_pos = np.append(hist['train_accuracy_false_pos'], args['train_accuracy_false_pos'].cpu()/args['len_training_ids'])
    train_accuracy_false_neg = np.append(hist['train_accuracy_false_neg'], args['train_accuracy_false_neg'].cpu()/args['len_training_ids'])

    validation_accuracy_true_pos = np.append(hist['validation_accuracy_true_pos'],args['validation_accuracy_true_pos'].cpu()/args['len_validation_ids'])
    validation_accuracy_true_neg = np.append(hist['validation_accuracy_true_neg'],args['validation_accuracy_true_neg'].cpu()/args['len_validation_ids'])
    validation_accuracy_false_pos = np.append(hist['validation_accuracy_false_pos'], args['validation_accuracy_false_pos'].cpu()/args['len_validation_ids'])
    validation_accuracy_false_neg = np.append(hist['validation_accuracy_false_neg'], args['validation_accuracy_false_neg'].cpu()/args['len_validation_ids'])


    np.savez(args['save_dir_model']+"/History", time_stamp=args['time_stamp'],
             train_loss= train_loss,
             train_accuracy = train_accuracy,
             train_accuracy_false_neg = train_accuracy_false_neg,
             train_accuracy_false_pos = train_accuracy_false_pos,
             train_accuracy_true_pos = train_accuracy_true_pos,
             train_accuracy_true_neg = train_accuracy_true_neg,
             validation_loss = validation_loss,
             validation_accuracy = validation_accuracy,
             validation_accuracy_true_pos = validation_accuracy_true_pos,
             validation_accuracy_true_neg = validation_accuracy_true_neg,
             validation_accuracy_false_pos = validation_accuracy_false_pos,
             validation_accuracy_false_neg = validation_accuracy_false_neg)

    if len(np.append(validation_loss,args['validation_loss']))>0:
        if len(np.append(validation_loss,args['validation_loss']))-np.argmin(np.append(validation_loss,args['validation_loss']))>100:
            sys.exit()

    converged = 0
    if len(np.append(validation_loss,args['validation_loss']))>0:
        if len(np.append(validation_loss,args['validation_loss']))-np.argmin(np.append(validation_loss,args['validation_loss']))>10:
            converged  = 1

    return converged