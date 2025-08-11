# ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Models
**(Accepted) [CoRL 2025]** This repository is the official repository for **ImLPR** **[[Paper]](https://arxiv.org/abs/2505.18364)** **[[Video]](https://youtu.be/8t-hVO4yPdg)**.

  <a href="https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko" target="_blank">Minwoo Jung</a><sup></sup>,
  <a href="https://scholar.google.com/citations?user=fqfPCUkAAAAJ&hl=ko" target="_blank">Lanke Frank Tarimo Fu</a><sup></sup>,
  <a href="https://scholar.google.com/citations?user=BqV8LaoAAAAJ&hl=ko" target="_blank">Maurice Fallon</a><sup></sup>,
  <a href="https://scholar.google.co.kr/citations?user=7yveufgAAAAJ&hl=ko" target="_blank">Ayoung Kim</a><sup>†</sup>

Collaboration with **[Robust Perception and Mobile Robotics Lab (RPM)](https://rpm.snu.ac.kr/)** and **[Dynamic Robot Systems Group (DRS)](https://dynamic.robots.ox.ac.uk/)**

The code will be realeased soon!

### Recent Updates
- [2025/08/11] First release of ImLPR repository! 

### Contributions
Our work makes the following contributions:
1. **ImLPR is the first LPR pipeline using a VFM while retaining the majority of pre-trained knowledge**: Our key innovation lies in a tailored three-channel RIV representation and lightweight convolutional adapters, which seamlessly bridge the 3D LiDAR and 2D vision domain gap. Freezing most DINOv2 layers preserves pre-trained knowledge during training, ensuring strong generalization and outperforming task-specific LPR networks.
2. **We introduce the Patch-InfoNCE loss**: A patch-level contrastive loss to enhance the local discriminability and robustness of learned LiDAR features. We demonstrate that our patch-level contrastive learning strategy achieves a performance boost in LPR.
3. **ImLPR demonstrates versatility on multiple public datasets**: Outperforming SOTA methods. Furthermore, we also validate the importance of each component of the ImLPR pipeline, with code available post-review for robotics community integration.
   
<img width="2194" height="735" alt="Selection_1878" src="https://github.com/user-attachments/assets/b8172090-987c-4a2a-9c50-af164148ea69" />
