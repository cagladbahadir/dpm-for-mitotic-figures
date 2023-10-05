import torch
import torch.nn as nn
import os


def saver(prediction, epoch,save_dir_model,mite_or_not):
    epoch = str(epoch)
    if not os.path.isdir(save_dir_model+'/seed_'+epoch): os.makedirs(save_dir_model+'/seed_'+epoch)
    for i in range(len(prediction)):
        torch.save(prediction[i], save_dir_model+'/seed_'+epoch+"/score_"+mite_or_not+".pt")

def forward_pass(net,input2,sigm):

    output = net(input2)
    prediction = sigm(output)
    return prediction, output

def loss_calculator( prediction,all_labels,loss_function): #

    if loss_function == 'BCE':
        criterion = nn.BCELoss()
    loss = criterion(prediction[:, 0], all_labels[:, 0])

    return loss


def accuracy_calculator(prediction,labels):
    test_accuracy = torch.sum((torch.round(prediction[:,0]) == torch.round(labels[:,0])).float())/  torch.tensor(len(labels))
    true_pos = torch.sum(
        (torch.round(prediction[labels[:, 0] >= 0.5, 0]) == torch.round(labels[labels[:, 0] >= 0.5, 0]))) / torch.sum(
        (labels[:, 0] >= 0.5).float())
    true_neg = torch.sum(
        (torch.round(prediction[labels[:, 0] < 0.5, 0]) == torch.round(labels[labels[:, 0] < 0.5, 0]))) / torch.sum(
        (labels[:, 0] < 0.5).float())

    false_pos = torch.sum((labels[torch.round(prediction[:,0])==1.,0] < 0.5).float())/(0.005+torch.sum(torch.round(prediction[torch.round(prediction[:,0])==1.,0]) == 1.).float())
    false_neg = torch.sum((labels[torch.round(prediction[:,0])==0.,0] >=0.5).float())/(0.005+torch.sum(torch.round(prediction[torch.round(prediction[:,0])==0.,0]) == 0.).float())
    return test_accuracy, true_pos, true_neg, false_pos,false_neg

