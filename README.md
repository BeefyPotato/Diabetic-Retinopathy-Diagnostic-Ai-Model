# **Making AI Accessible for Medical Diagnostics**

Artificial intelligence is transforming industries, yet its reliance on large datasets and expensive computational resources often limits its adoption. This challenge is especially pronounced in the **medical industry**, where data collection is costly, constrained by privacy regulations, and under-resourced clinics often lack access to high-performance hardware. To address these challenges, we focused on creating an AI solution that emphasizes **efficiency, accessibility, and scalability**.

This project fine-tunes the **Swin-S Transformer** to classify the severity of diabetic retinopathy (DR) using a **minimal dataset of 25,000 images**. Training was completed in just **546 steps (2 epochs)** on a **T4 GPU** in under **30 minutes** using **Google Cloud Platform (GCP)**—a resource freely available on **Google Colab**. Despite the constrained setup, the model achieves **86.9% accuracy** in detecting DR and **70% accuracy** in stage-specific classification.

---

## **Key Highlights**
- **Minimal Resources, Maximum Impact**: The model delivers competitive performance while being trained on a small dataset and minimal compute resources, making it ideal for settings with limited infrastructure.
- **Scalable and Accessible**: Designed for training on hardware as accessible as free Google Colab GPUs, this solution is practical for deployment in under-resourced hospitals and clinics.
- **Real-World Applicability**: Achieves clinically relevant metrics for DR detection, enabling early diagnosis and improved patient outcomes in constrained environments.

This project demonstrates the potential for **cost-effective and scalable diagnostic tools** in the medical industry. By showcasing how state-of-the-art AI can achieve exceptional results with limited data and compute power, we bridge the gap between innovation and real-world impact, paving the way for equitable AI-driven healthcare solutions.

---

## **Dataset**
- **Size**: 25,000 images (224x224 pixels) evenly distributed across the 5 stages of DR:
  - **Stages**: None, Mild, Moderate, Severe, Proliferative.
- **Augmentations**: Two augmentations applied to each image for robustness.
- **Split**: 
  - Training: 17,500 images
  - Validation: 3,750 images
  - Test: 3,750 images
- **Labels**: Each image is labeled from 0 to 4, corresponding to the DR stage.

---

## **Model**
The Swin Transformer effectively bridges the gap between **CNNs** and **Vision Transformers (ViTs)**:
- **CNNs** excel at capturing local features but often struggle with global context.
- **ViTs** are strong in modeling global dependencies but lack efficiency with high-resolution medical images.
  
The **Swin Transformer’s hierarchical architecture** and **shifted window attention mechanism** address both limitations, capturing **fine-grained details** and **global patterns** efficiently.  
- **Architecture**: Swin-B Transformer with a reinitialized final linear layer (`out_features=5`).
- **Pretrained Weights**: Retrieved from the Swin Transformer GitHub repository.

---

## **Training Setup**
- **Steps**: 546 steps (2 epochs).
- **Batch Size**: 64.
- **Learning Rate**:
  - **Base**: `1e-4`, with cosine annealing after a 50-step warmup.
  - **Head**: `1e-3` for faster learning and stronger regularization.
- **Weight Decay**:
  - **Base**: `1e-4`.
  - **Head**: `1e-3`.
- **Drop Rate**: 0.0 for minimal regularization.

---

## **Performance Metrics**
We evaluated the model using two types of metrics:
### **Severity Metrics**
1. **Cross-Entropy Loss**
2. **Accuracy**
3. **Prediction Distance**:
   - Measures how far predictions are from the true label:
     - **Distance 1**: One stage away (e.g., predicting "Moderate" instead of "Severe").
     - **Distance 2–4**: Greater deviations, reflecting the clinical repercussions of incorrect predictions.
   
### **Binary Metrics**
- Evaluates DR presence (index 0 for absent, indices 1–4 for present).
  
---

## **Results**
### **Training Metrics**
- **Training Loss (Initial)**: ~1.61 (consistent with expected loss of ln(out_features) = ln(5) = 1.609)

  
### **Validation Metrics (Final)**:
- **Loss**: 0.658
- **Accuracy (Classification)**: 73.8%
- **Accuracy (Binary)**: 87.3%
- **Prediction Distances**:
  - Distance 1: 16.6%
  - Distance 2: 9.2%
  - Distance 3: 0.3%
  - Distance 4: 0.1%

---

## **Running the Training Job**
1. Navigate to `~/diabetic_ai/model`.
2. Create a Determined AI experiment:
   ```bash
   det experiment create ~/diabetic_ai/model/config.yaml ~/diabetic_ai/model
   ```
3. Edit hyperparameters in `config.yaml` as needed.  
   - Set the total epochs under the `searcher` section.


## Acknowledgments

This project uses the [Swin-S Transformer](https://arxiv.org/abs/2103.14030) architecture. The pretrained model weights were obtained from [Microsoft's Swin Transformer GitHub repository](https://github.com/microsoft/Swin-Transformer).

Citation for the original Swin Transformer paper:
@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}

