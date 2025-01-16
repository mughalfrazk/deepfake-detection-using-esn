# Eye-Movement Based Deepfake Detection Using Reservoir Computing  

# Abstract
The increasing sophistication of deepfake generation technologies poses significant
challenges to digital media authenticity, undermining trust in visual content used across
various domains, including legal investigations and news reporting. Deepfake videos,
created using advanced techniques such as Generative Adversarial Networks (GANs),
have become increasingly difficult to distinguish from genuine content.
<br /><br />
This study proposes a novel deepfake detection approach that leverages biometric and
media-based features, with a particular focus on eye-blinking patterns. By analyzing
temporal anomalies in eye movements, we aim to uncover traces of manipulation that
remain undetected by conventional methods. Using the FaceForensics++ dataset, which
includes both real and fake videos, we preprocessed the videos by detecting faces,
extracting eye-blinking dynamics (Eye Aspect Ratio, delta EAR), and leveraging the
ResNet50 model to generate high-dimensional feature representations. These features
were then reduced using Principal Component Analysis (PCA) and combined with
biometric features to form a 53-dimensional feature vector per frame.
<br /><br />
The experimentation includes a comprehensive comparison between the training
outcomes using different hyperparameter configurations for the Echo State Network
(ESN) model. Various hyperparameters, such as the reservoir size, spectral radius, leaking
rate, and the number of neurons in the output layer, were systematically tested to
evaluate their impact on the model’s performance. Each configuration was assessed
based on its ability to capture the temporal patterns of eye blinking and movement, with
the goal of maximizing classification accuracy and minimizing false positives. By
comparing the results of these different training setups, the study aimed to identify the
optimal set of hyperparameters that would improve the model’s ability to distinguish
between real and deepfake videos, thus enhancing the overall effectiveness of the
deepfake detection system.
<br /><br />
The ESN model was trained and evaluated, achieving a classification accuracy of 67%.
Despite this moderate performance, the results indicate that ESNs can capture temporal
dependencies in eye movements, which are essential for differentiating deepfake and
real videos. The model exhibited challenges in generalizing between classes, with false
positives being more common. To improve performance, future directions include
enhancing feature extraction by incorporating additional facial dynamics, exploring
alternative temporal models such as LSTMs or GRUs, and experimenting with advanced
data augmentation techniques. Furthermore, using a more focused dataset better suited
for extracting eye blinks could improve the model’s discriminative power. The source
code is for the project is publicly available in the repository:
https://github.com/mughalfrazk/deepfake-detection-using-esn

## Overview  
This project is part of my MSc dissertation at Sheffield Hallam University, focusing on deepfake detection using biometric and media-based features. By analyzing temporal anomalies in eye-blinking patterns, this research leverages **Echo State Networks (ESNs)** to distinguish real videos from deepfakes.  

The project combines advanced AI techniques with biometric analysis using eye-blinking patterns to propose a novel, efficient approach to deepfake detection.

![Real time EAR Detection](https://github.com/mughalfrazk/deepfake-detection-using-esn/blob/main/assets/realtime-ear.gif?raw=true)

## Key Features  
- **Eye-Blinking Analysis**: Utilizes temporal patterns of eye blinks as a key biometric feature.  
- **Deep Learning Integration**: Extracts high-dimensional features using **ResNet50** and reduces them using **PCA**.  
- **Reservoir Computing**: Implements Echo State Networks (ESNs) for efficient and scalable temporal sequence modeling.  
- **Dataset**: Built and trained on the **FaceForensics++** dataset.  
<br />

![Feature Extraction](https://github.com/mughalfrazk/deepfake-detection-using-esn/blob/main/assets/feature-extraction.png?raw=true)

## Methodology  
1. **Preprocessing**:  
   - Face detection and landmark extraction using **MediaPipe**.  
   - Calculation of Eye Aspect Ratio (EAR) and Delta EAR to capture blinking behavior.  
   - High-dimensional feature extraction using **ResNet50** and dimensionality reduction with **PCA**.  
2. **Model Training**:  
   - ESN architecture optimized for sequential data, focusing on temporal dependencies in eye movement.  
   - Hyperparameter tuning for improved accuracy.  
3. **Evaluation**:  
   - Achieved a classification accuracy of 67%.  
   - Analyzed false positives and areas for improvement.  

## Results  
The ESN model demonstrated potential for capturing temporal dependencies in eye movements, though challenges remain in improving generalization and accuracy.  

## Future Directions  
- Enhance feature extraction by incorporating additional facial dynamics (e.g., lip and eyebrow movements).  
- Explore alternative temporal models like **LSTMs**, **GRUs**, or Transformer-based architectures.  
- Improve dataset quality to focus more on eye-blinking dynamics.  


