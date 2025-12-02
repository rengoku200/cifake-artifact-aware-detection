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




---Trained baseline summary 
Using device: cpu
Epoch 1/2
Train loss: 0.2836 | Train acc: 0.8762                                                                                
Val   loss: 0.1884 | Val   acc: 0.9229

Epoch 2/2
Train loss: 0.1630 | Train acc: 0.9362                                                                                
Val   loss: 0.1589 | Val   acc: 0.9377