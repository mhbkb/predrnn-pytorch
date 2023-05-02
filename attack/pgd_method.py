import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.datasets as dsets
import torchvision.transforms as transforms

def pgd_attack(model, images, mask, labels, device, eps=0.1, alpha=2/255, iters=200) :
    images = images.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    loss = nn.MSELoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        # import pdb; pdb.set_trace()
        images.requires_grad = True
        
        outputs = model(images, mask)[0]
        outputs = outputs[:, -10:, :, :, :]
        
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images