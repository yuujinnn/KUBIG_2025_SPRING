# **3D Reconstruction of Independence Activists Using Stable Diffusion**  
Generating 3D models of historical independence activists utilizing pretrained diffusion models.

---

## **Overview**  
This project aims to reconstruct **3D models** of Korean independence activists using diffusion-based generative models. By **fine-tuning** a pretrained **Stable-DreamFusion** model, we generate realistic **3D representations** from limited 2D images.

### **Cited Works**  
- **Stable-DreamFusion** [(GitHub)](https://github.com/ashawkey/stable-dreamfusion)  
- **DCO Fine-tuning Methodology** [(GitHub)](https://github.com/kyungmnlee/dco)  
- **Our Works of Implementing Fine-tuned Model to DreamFusion** [(GitHub)](https://github.com/wltschmrz/stable_dreamfusion_deprecated)  
---

## **Pipeline**  

### **1. Data Collection for Fine-tuning**  
We gathered **historical images** of Korean independence activists for model training. These images were preprocessed to enhance clarity and usability.  

**Dataset Preview:**  
📁 `/data/independence_activists/`  
- `/path/to/image1.jpg`  
- `/path/to/image2.jpg`  
- `/path/to/image3.jpg`  

---

### **2. Fine-tuning with DCO**  
Using **DCO (DreamFusion Control Optimization)**, we adapted the diffusion model to better represent historical figures in a **3D-consistent** manner.  

**Fine-tuning Samples:**  
| Input Image | Generated Output | Caption |
|------------|----------------|---------|
| ![Sample 1](sample1.jpg) | ![Output 1](output1.jpg) | "Reconstruction of [Activist Name]" |
| ![Sample 2](sample2.jpg) | ![Output 2](output2.jpg) | "Generated face from historical dataset" |

---

### **3. 3D Sampling with DreamFusion**  
By leveraging **Stable-DreamFusion**, we reconstructed **3D volumetric samples** from the fine-tuned model. The results demonstrate a high level of consistency in identity preservation.

**Generated 3D Samples:**  
| Example 1 | Example 2 | Example 3 |
|-----------|-----------|-----------|
| ![3D Sample 1](sample1.gif) | ![3D Sample 2](sample2.gif) | ![3D Sample 3](sample3.gif) |

---

### **4. 3D Model Generation from Fine-tuned 2D Diffusion Model**  
Using the **fine-tuned 2D model**, we generated **high-fidelity 3D models** of independence activists. The results showcase realistic depth, shading, and facial structures.

**Final 3D Model Outputs:**  
📁 `/results/3D_models/`  
- `activist1.obj`  
- `activist2.obj`  
- `activist3.obj`  

---

## **Installation & Usage**  

### **Setup**
Clone the repository and install dependencies:
```bash
git clone https://github.com/wltschmrz/stable_dreamfusion_deprecated.git
cd stable_dreamfusion_deprecated
pip install -r requirements.txt
