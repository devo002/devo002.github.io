---
layout: single
title: "David Onaiyekan"
permalink: /
author_profile: false   # hide the big name card on the left
---

## Machine Learning & AI Engineer

I'm a Machine Learning & AI Engineer with a focus on **computer vision**, **generative models**, and **MLOps**.  
I enjoy taking projects from research ideas to stable, production-ready systems.

---

## Skills

**Machine Learning & Deep Learning**

- Supervised and semi-supervised learning (FixMatch, MixMatch)  
- CNNs, transformers, sequence models  
- Model evaluation, cross-validation, hyperparameter tuning (Optuna)

**Computer Vision & Generative Models**

- Defect detection, handwriting recognition, object proposals  
- GAN-based and diffusion-based handwriting and style generation  
- OpenCV, image preprocessing, data augmentation

**MLOps & Engineering**

- FastAPI, REST APIs, Streamlit / Gradio apps  
- Docker, basic CI/CD, deploying to cloud (AWS / other providers)  
- Working with Linux/Windows servers, networking, and system administration

**Tools & Libraries**

- Python, PyTorch, TensorFlow, scikit-learn  
- NumPy, Pandas, Matplotlib, OpenCV  
- Git, GitHub, VS Code, Jupyter
---

## Featured Projects {#projects}

### 1. Semi-Supervised Defect Detection in Coil Winding

Built a semi-supervised pipeline using **FixMatch** on **DINOv2** and **EfficientNetV2** backbones to classify defects in coil winding images with very few labeled samples.  
Achieved strong **macro F1** scores and used **Optuna** to tune hyperparameters.

> **Tech:** PyTorch, DINOv2, EfficientNetV2, FixMatch, MixMatch, Optuna, Computer Vision  
> **Role:** End-to-end design, training, evaluation, reporting

---

### 2. Handwriting Generation with AFFGAN & Diffusion Models

Reproduced and extended the **AFFGANwriting** framework for handwriting style generation.  
Explored replacing the original VGG backbone with **EfficientNet** and **DINOv2** and experimented with **diffusion models** as alternative generators to improve visual quality and FID.

> **Tech:** PyTorch, GANs, Diffusion Models, EfficientNet, DINOv2  
> **Focus:** Model architecture, style representation, qualitative and quantitative evaluation

---

### 3. Handwriting Recognition (ConTran-based Pipeline)

Implemented an encoder–decoder handwriting recognition system with **CNN + RNN + Attention**.  
Evaluated different encoders (**VGG19**, **EfficientNetV2**, **ResNet50**) and applied **Grad-CAM** to interpret recognition errors and visualize attention over words.

> **Tech:** PyTorch, CNN/RNN, Attention, Grad-CAM  
> **Focus:** Architecture comparison, error analysis, visualization

---

### 4. Selective Search Object Proposal System

Implemented a custom **Selective Search** algorithm for object proposals: image segmentation, region similarity metrics, and hierarchical merging.  
Evaluated proposal quality on COCO-style datasets.

> **Tech:** Python, OpenCV, NumPy, Matplotlib  
> **Focus:** Classical computer vision, algorithm implementation, evaluation

---

If you’d like to know more about any of these projects or collaborate, feel free to reach out at  
📧 **davidmayowaonaiyekan@gmail.com** or connect on [LinkedIn](https://www.linkedin.com/in/david-mayowa-onaiyekan-01b436122).
