import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import config
from model import Generator,Discriminator
import utils
import dataset
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader

def train(train_data,gen_model,dis_model,gen_optimiser,dis_optimiser,epochs, batch_size,lr):
    gen_loss=[]
    dis_loss=[]
    gen_acc=[]
    dis_acc=[]
    
    
    loss_func = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()
    
    for epoch in range(epochs):
        g_loss=[]
        d_loss=[]
        #x is sketch image and y is coloured image
        for idx, (x,y) in enumerate(train_data):
            x=x.to(config.DEVICE)
            y=y.to(config.DEVICE)
            ### train discriminator 
            D_real = dis_model(x,y)
            real_targets=torch.ones_like(D_real).to(config.DEVICE)
            # print(D_real.shape, real_targets.shape)
            D_real_loss = loss_func(D_real,real_targets)
            y_fake = gen_model(x)
            D_fake = dis_model(x,y_fake)
            fake_targets=torch.zeros_like(D_fake).to(config.DEVICE)
            D_fake_loss = loss_func(D_fake,fake_targets)
            D_loss = (D_real_loss + D_fake_loss)/2
            dis_model.zero_grad()
            d_loss.append(D_loss.item())
            D_loss.backward(retain_graph=True)
            dis_optimiser.step()

            ### train generator
            D_fake = dis_model(x,y_fake)
            G_fake_loss = loss_func(D_fake, torch.ones_like(D_fake).to(config.DEVICE))
            L1loss = L1_loss(y_fake,y)*config.LAMBDA1
            # print(type(G_fake_loss),type(L1_loss))
            G_loss = G_fake_loss + L1loss
            gen_model.zero_grad()
            g_loss.append(G_loss.item())
            G_loss.backward()
            gen_optimiser.step()

        print(f"Epoch [{epoch}/{epochs}] \Loss D: {d_loss[-1]:.4f}, loss G: {g_loss[-1]:.4f}")

        if config.SAVE_MODEL and epoch % 10 == 0:
            utils.save_checkpoint(gen_model, gen_optimiser, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(dis_model, dis_optimiser, filename=config.CHECKPOINT_DISC)

        # if epoch%5 == 0:
        utils.save_some_examples(gen_model, val_loader,epoch,folder="/DATA/bhumika1/Documents/Nikhil/pix2pix/results")


if __name__=='__main__':
    path = "/DATA/bhumika1/Documents/Nikhil/pix2pix/data/data/"
    train_dataset = dataset.AnimeDataset(path,'train')
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY)
    val_dataset = dataset.AnimeDataset(path,'val')
    val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY)
    gen_model = Generator(in_channels=3,out_channels=64)
    dis_model = Discriminator()
    gen_optimiser=torch.optim.Adam(gen_model.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    dis_optimiser=torch.optim.Adam(dis_model.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    ## load the saved model
    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen_model,gen_optimiser,config.LEARNING_RATE)
        utils.load_checkpoint(config.CHECKPOINT_DISC,dis_model,dis_optimiser,config.LEARNING_RATE)

    gen_model = gen_model.to(config.DEVICE)
    dis_model = dis_model.to(config.DEVICE)
    ### train function
    gen_optimiser=torch.optim.Adam(gen_model.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    dis_optimiser=torch.optim.Adam(dis_model.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    train(train_loader,gen_model,dis_model,gen_optimiser,dis_optimiser,config.EPOCHS,config.BATCH_SIZE,config.LEARNING_RATE)


    
