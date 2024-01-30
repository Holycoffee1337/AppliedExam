# PROBLEM 1

## Table of Contents
1. [Introduction](#introduction)
2. [Preconditions](#preconditions)
3. [Usage](#usage)
4. [Files Description](#files-description)

## Introduction
This primary objective of this problem, is to perform image classification on the PCam dataset, with a focus on using autoencoders for image compression. Link to PCAM "https://syddanskuni-my.sharepoint.com/:f:/g/personal/cmd_sam_sdu_dk/EiWD2LmuxCJBp-_tfGK7aL8Bdt2cPsb6MCpVs1pOYUcXAw?email=cmd%40sam.sdu.dk&e=2Vx6tL" 

## Precondtions
- The data saved, is required to be saved in the folder `'./data/'`
- When training the CNN using AE or VAE, it is required, to have saved model locally to compress the image

## Usage
To train and run AE`
model = TrainAE()
model.train(3)
model.run(10)` 

To train and run VAE`
model = TrainVAE()
model.train(3)
model.run(10)
`

To CNN Resize
`
model_cnn = TrainCNN()
model_cnn.train(5)
`

## File Descriptions

- data.py: File handsle the data processing
- 
