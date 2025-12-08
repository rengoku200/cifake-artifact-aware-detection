Setup---
python3 -m venv venv (paste into terminal)
source venv/bin/activate (activate the envrionment)
pip install --upgrade pip
Installs---
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install pillow numpy matplotlib tqdm scikit-learn captum




Things to do:

1. Set up environment
Create virtual environment (python3 -m venv venv)
Install required packages (torch, torchvision, pillow, numpy, etc.)

2. Download and prepare CIFAKE dataset
Download dataset from Kaggle by running downloadImage.py
Place into: ../data folder

3. Implement PyTorch dataset + dataloader
Write CIFAKEDataset class
Confirm loads images correctly using data.py

4. Build & train the baseline CNN
Implement BaselineCNN model
Train for a few epochs
Evaluate accuracy

5. Implement upgraded architecture (our contribution)

Our new model adds:
Frequency-domain branch (FFT/DCT of images)
Spatial RGB branch (normal CNN)
Attention mechanism (CBAM or simple attention block)
Fusion layer that combines both branches

6. Train upgraded architecture
Train same way as baseline
Compare metrics (accuracy, loss, ROC-AUC)

7. Add interpretability
Use advanced methods:
Integrated Gradients
Occlusion Sensitivity
Frequency masking / ablation


8. Evaluate generalization
Test your models on new synthetic images not in CIFAKE
(e.g., Stable Diffusion v2, DALL·E mini)

9. Compare all models

Compare:
Baseline CNN
Pretrained models (ResNet, VGG, etc.)
Your upgraded frequency + attention model

Metrics:
Accuracy
Loss
Confusion matrix
Interpretability visualizations
Cross-generator performance

10. Final report + GitHub cleanup

Generate plots
Include interpretability examples
Summarize findings
Document code in README




---Trained baseline final summary 
Using device: mps
Training BaselineCNN...

Epoch 1/10
Train loss: 0.3225 | Train acc: 0.8553                                                                                      
Val   loss: 0.2178 | Val   acc: 0.9091

Epoch 2/10
Train loss: 0.1812 | Train acc: 0.9290                                                                                      
Val   loss: 0.1809 | Val   acc: 0.9286

Epoch 3/10
Train loss: 0.1417 | Train acc: 0.9444                                                                                      
Val   loss: 0.1494 | Val   acc: 0.9421

Epoch 4/10
Train loss: 0.1127 | Train acc: 0.9563                                                                                      
Val   loss: 0.2070 | Val   acc: 0.9235

Epoch 5/10
Train loss: 0.0877 | Train acc: 0.9663                                                                                      
Val   loss: 0.1823 | Val   acc: 0.9338

Epoch 6/10
Train loss: 0.0652 | Train acc: 0.9751                                                                                      
Val   loss: 0.1743 | Val   acc: 0.9436

Epoch 7/10
Train loss: 0.0482 | Train acc: 0.9818                                                                                      
Val   loss: 0.2117 | Val   acc: 0.9390

Epoch 8/10
Train loss: 0.0371 | Train acc: 0.9864                                                                                      
Val   loss: 0.2138 | Val   acc: 0.9415

Epoch 9/10
Train loss: 0.0293 | Train acc: 0.9891                                                                                      
Val   loss: 0.2462 | Val   acc: 0.9416

Epoch 10/10
Train loss: 0.0261 | Train acc: 0.9908                                                                                      
Val   loss: 0.2552 | Val   acc: 0.9381

------------
trained paper summary:

Epoch 1/12
Train loss: 0.3238 | Train acc: 0.8571                                                                                      
Val   loss: 0.2335 | Val   acc: 0.9025

Epoch 2/12
Train loss: 0.2066 | Train acc: 0.9167                                                                                      
Val   loss: 0.1909 | Val   acc: 0.9228

Epoch 3/12
Train loss: 0.1782 | Train acc: 0.9291                                                                                      
Val   loss: 0.1815 | Val   acc: 0.9283

Epoch 4/12
Train loss: 0.1580 | Train acc: 0.9381                                                                                      
Val   loss: 0.1767 | Val   acc: 0.9284

Epoch 5/12
Train loss: 0.1453 | Train acc: 0.9431                                                                                      
Val   loss: 0.1901 | Val   acc: 0.9250

Epoch 6/12
Train loss: 0.1339 | Train acc: 0.9482                                                                                      
Val   loss: 0.1539 | Val   acc: 0.9399

Epoch 7/12
Train loss: 0.1230 | Train acc: 0.9527                                                                                      
Val   loss: 0.1487 | Val   acc: 0.9420

Epoch 8/12
Train loss: 0.1121 | Train acc: 0.9566                                                                                      
Val   loss: 0.1465 | Val   acc: 0.9428

Epoch 9/12
Train loss: 0.1019 | Train acc: 0.9605                                                                                      
Val   loss: 0.1469 | Val   acc: 0.9438

Epoch 10/12
Train loss: 0.0940 | Train acc: 0.9636                                                                                      
Val   loss: 0.1728 | Val   acc: 0.9347

Epoch 11/12
Train loss: 0.0862 | Train acc: 0.9668                                                                                      
Val   loss: 0.1628 | Val   acc: 0.9426

Epoch 12/12
Train loss: 0.0760 | Train acc: 0.9713                                                                                      
Val   loss: 0.1624 | Val   acc: 0.9448
-----------------
trained artifact aware summary:
Epoch 1/12
Train loss: 0.3238 | Train acc: 0.8571                                                                                      
Val   loss: 0.2335 | Val   acc: 0.9025

Epoch 2/12
Train loss: 0.2066 | Train acc: 0.9167                                                                                      
Val   loss: 0.1909 | Val   acc: 0.9228

Epoch 3/12
Train loss: 0.1782 | Train acc: 0.9291                                                                                      
Val   loss: 0.1815 | Val   acc: 0.9283

Epoch 4/12
Train loss: 0.1580 | Train acc: 0.9381                                                                                      
Val   loss: 0.1767 | Val   acc: 0.9284

Epoch 5/12
Train loss: 0.1453 | Train acc: 0.9431                                                                                      
Val   loss: 0.1901 | Val   acc: 0.9250

Epoch 6/12
Train loss: 0.1339 | Train acc: 0.9482                                                                                      
Val   loss: 0.1539 | Val   acc: 0.9399

Epoch 7/12
Train loss: 0.1230 | Train acc: 0.9527                                                                                      
Val   loss: 0.1487 | Val   acc: 0.9420

Epoch 8/12
Train loss: 0.1121 | Train acc: 0.9566                                                                                      
Val   loss: 0.1465 | Val   acc: 0.9428

Epoch 9/12
Train loss: 0.1019 | Train acc: 0.9605                                                                                      
Val   loss: 0.1469 | Val   acc: 0.9438

Epoch 10/12
Train loss: 0.0940 | Train acc: 0.9636                                                                                      
Val   loss: 0.1728 | Val   acc: 0.9347

Epoch 11/12
Train loss: 0.0862 | Train acc: 0.9668                                                                                      
Val   loss: 0.1628 | Val   acc: 0.9426

Epoch 12/12
Train loss: 0.0760 | Train acc: 0.9713                                                                                      
Val   loss: 0.1624 | Val   acc: 0.9448
Saved paper model to ../checkpoints/paper_cnn.pt
(venv) (base) baidymbaye@Baidys-MacBook-Pro-2982 src % python train_models.py
Using Apple Silicon GPU (MPS)
Device: mps
Training ArtifactAwareCNN...

Epoch 1/20
Train loss: 0.3526 | Train acc: 0.8295                                                                                      
Val   loss: 0.2171 | Val   acc: 0.9126

Epoch 2/20
Train loss: 0.2077 | Train acc: 0.9163                                                                                      
Val   loss: 0.1985 | Val   acc: 0.9193

Epoch 3/20
Train loss: 0.1835 | Train acc: 0.9274                                                                                      
Val   loss: 0.1750 | Val   acc: 0.9297

Epoch 4/20
Train loss: 0.1676 | Train acc: 0.9341                                                                                      
Val   loss: 0.1592 | Val   acc: 0.9366

Epoch 5/20
Train loss: 0.1607 | Train acc: 0.9371                                                                                      
Val   loss: 0.1589 | Val   acc: 0.9383

Epoch 6/20
Train loss: 0.1498 | Train acc: 0.9422                                                                                      
Val   loss: 0.1522 | Val   acc: 0.9399

Epoch 7/20
Train loss: 0.1413 | Train acc: 0.9451                                                                                      
Val   loss: 0.1745 | Val   acc: 0.9323

Epoch 8/20
Train loss: 0.1374 | Train acc: 0.9464                                                                                      
Val   loss: 0.1514 | Val   acc: 0.9395

Epoch 9/20
Train loss: 0.1310 | Train acc: 0.9495                                                                                      
Val   loss: 0.1305 | Val   acc: 0.9496

Epoch 10/20
Train loss: 0.1266 | Train acc: 0.9515                                                                                      
Val   loss: 0.1433 | Val   acc: 0.9437

Epoch 11/20
Train loss: 0.1228 | Train acc: 0.9523                                                                                      
Val   loss: 0.1329 | Val   acc: 0.9481

Epoch 12/20
Train loss: 0.1170 | Train acc: 0.9547                                                                                      
Val   loss: 0.1245 | Val   acc: 0.9520

Epoch 13/20
Train loss: 0.1160 | Train acc: 0.9552                                                                                      
Val   loss: 0.1229 | Val   acc: 0.9516

Epoch 14/20
Train loss: 0.1108 | Train acc: 0.9577                                                                                      
Val   loss: 0.1387 | Val   acc: 0.9473

Epoch 15/20
Train loss: 0.1080 | Train acc: 0.9588                                                                                      
Val   loss: 0.1413 | Val   acc: 0.9448

Epoch 16/20
Train loss: 0.1029 | Train acc: 0.9611                                                                                      
Val   loss: 0.1417 | Val   acc: 0.9432

Epoch 17/20
Train loss: 0.1022 | Train acc: 0.9611                                                                                      
Val   loss: 0.1228 | Val   acc: 0.9528

Epoch 18/20
Train loss: 0.0987 | Train acc: 0.9623                                                                                      
Val   loss: 0.1438 | Val   acc: 0.9444

Epoch 19/20
Train loss: 0.0959 | Train acc: 0.9629                                                                                      
Val   loss: 0.1261 | Val   acc: 0.9516


Training VGG16CIFAKE...

Epoch 1/5
Train loss: 0.2591 | Train acc: 0.8997                                                                                  
Val   loss: 0.1607 | Val   acc: 0.9452

Epoch 2/5
Train loss: 0.1613 | Train acc: 0.9423                                                                                  
Val   loss: 0.1385 | Val   acc: 0.9491

Epoch 3/5
Train loss: 0.1203 | Train acc: 0.9575                                                                                  
Val   loss: 0.1332 | Val   acc: 0.9523

Epoch 4/5
Train loss: 0.0978 | Train acc: 0.9668                                                                                  
Val   loss: 0.1389 | Val   acc: 0.9532

Epoch 5/5
Train loss: 0.0820 | Train acc: 0.9730                                                                                  
Val   loss: 0.1252 | Val   acc: 0.9563
