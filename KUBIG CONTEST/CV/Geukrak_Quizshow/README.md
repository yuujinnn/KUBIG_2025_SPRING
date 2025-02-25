# **3D Reconstruction of Independence Activists Using Stable Diffusion**  
Generating 3D models of historical independence activists utilizing pretrained T2I 2D diffusion models.

---

## **Overview**  
This project aims to reconstruct **3D models** of Korean independence activists using diffusion-based generative models. By **fine-tuning** a pretrained **Stable-DreamFusion** model, we generate realistic **3D representations** from limited 2D images.
- **Our Works of Implementing Fine-tuned Model to DreamFusion** [(GitHub: wltschmrz)](https://github.com/wltschmrz/stable_dreamfusion_deprecated)  

### **Cited Works**  
- **Stable-DreamFusion** [(GitHub: https://github.com/ashawkey/stable-dreamfusion)](https://github.com/ashawkey/stable-dreamfusion)  
- **DCO Fine-tuning Methodology** [(GitHub: https://github.com/kyungmnlee/dco)](https://github.com/kyungmnlee/dco)  

---

## **Total Process**  

### **1. Data Collection for Fine-tuning**  
We gathered **historical images** of Korean independence activists for model training. These images were preprocessed to enhance clarity and usability.  

**Dataset Preview:**  
📁 `/results/prepared_datas.jpeg`
<p align="center">
  <img src="results/prepared_datas.jpeg" alt="data" width="400">
</p>

---

### **2. Fine-tuning with DCO**  
Using **DCO (Direct Consistency Optimization)**, we adapted a unique fine-tuning methodology for diffusion models to better represent historical facial figures with a limited amount of training data.

**Fine-tuning Samples:**

<table align="center">
  <tr>
    <th style="text-align:center;">Generated Output</th>
    <th style="text-align:center;">Text Caption</th>
  </tr>
  <tr>
    <td align="center">
      <img src="results/finetuned_sample_datas/test_front_42.png" width="100" style="display:block; margin:auto;">
      <img src="results/finetuned_sample_datas/test_side_42.png" width="100" style="display:block; margin:auto;">
      <img src="results/finetuned_sample_datas/test_back_42.png" width="100" style="display:block; margin:auto;">
      <img src="results/finetuned_sample_datas/test_plain_42.png" width="100" style="display:block; margin:auto;">
    </td>
    <td align="center">Reconstructions of [Changho_An]</td>
  </tr>
  <tr>
    <td align="center"><img src="results/finetuned_sample_datas/iter1000_A_DSLR_photo_of_mans_head_with_full_hair.jpeg" width="400" style="display:block; margin:auto;"></td>
    <td align="center">"A DSLR photo of [Changho_An]'s head with full hair"</td>
  </tr>
  <tr>
    <td align="center"><img src="results/finetuned_sample_datas/iter1000_A_DSLR_photo_of_mans_head_with_hair.jpeg" width="400" style="display:block; margin:auto;"></td>
    <td align="center">"A DSLR photo of [Changho_An]'s head with hair"</td>
  </tr>
  <tr>
    <td align="center"><img src="results/finetuned_sample_datas/iter1000_A_DSLR_photo_of_mans_head_with_hair_in_color.jpeg" width="400" style="display:block; margin:auto;"></td>
    <td align="center">"A DSLR photo of [Changho_An]'s head with hair in color"</td>
  </tr>
</table>

---

### **3. 3D Sampling with DreamFusion**  
By leveraging **Stable-DreamFusion**, we reconstructed **3D volumetric samples** from the 2D diffusion model. The generated 3D model maintains consistency across multiple perspectives, avoiding unnatural distortions.

**Generated 3D Samples:**  
| A_photo_of_a_burger | A_DSLR_photo_of_a_squirrel | A_DSLR_photo_of_a_bust | A_DSLR_photo_of_a_head |
|-----------|-----------|-----------|-----------|
| ![3D Sample 1](results/dreamfusion_samples/A_photo_of_a_burger.gif) | ![3D Sample 2](results/dreamfusion_samples/A_DSLR_photo_of_a_squirrel.gif) | ![3D Sample 3](results/dreamfusion_samples/A_DSLR_photo_of_a_bust.gif) | ![3D Sample 4](results/dreamfusion_samples/A_DSLR_photo_of_a_head_of_man.gif) |

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

pip uninstall torch torchvision torchaudio -y
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
sudo sh cuda_11.6.2_510.47.03_linux.run --toolkit --silent

echo 'export PATH=/usr/local/cuda-11.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-11.6' >> ~/.bashrc

source ~/.bashrc

echo $PATH
echo $LD_LIBRARY_PATH
nvcc --version

cd /workspace/stable-dreamfusion/gridencoder
python setup.py build_ext --inplace
```
Configure the pip dependencies:
```bash
huggingface_hub==0.19.4
diffusers==0.24.0
accelerate==0.19.0
transformers==4.30.2

numpy==1.24.4
```

---

## References  
- Stable-DreamFusion: [GitHub](https://github.com/ashawkey/stable-dreamfusion)  
- DCO(Direct Consistency Optimization): [GitHub](https://github.com/kyungmnlee/dco)  
