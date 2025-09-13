# Deep Learning Notebooks

A comprehensive collection of deep learning projects and coursework, featuring implementations from Coursera's Deep Learning Specialization and advanced machine learning challenges from ISE (Information Systems Engineering).

## 🎯 Overview

This repository showcases my journey through deep learning, from foundational concepts to advanced applications in computer vision, natural language processing, and cybersecurity. The projects demonstrate practical implementations of neural networks, ensemble methods, and state-of-the-art architectures.

## 📚 Coursera Deep Learning Specialization

### Course 1: Neural Networks and Deep Learning
- **Regularization Techniques**: Implementation of L1, L2, and dropout regularization
- **Gradient Checking**: Verification of gradient computations for neural networks
- **TensorFlow Introduction**: Hands-on experience with TensorFlow/Keras framework

### Course 2: Improving Deep Neural Networks
- **Hyperparameter Tuning**: Systematic approach to optimizing neural network performance
- **Optimization Algorithms**: Implementation of Adam, RMSprop, and other optimizers
- **Batch Normalization**: Understanding and implementing batch normalization layers

### Course 3: Convolutional Neural Networks
- **Residual Networks (ResNet)**: Implementation of skip connections and residual blocks
- **Transfer Learning**: Leveraging pre-trained models for image classification
- **Object Detection**: Introduction to YOLO and other detection algorithms

### Course 4: Convolutional Neural Networks (Advanced)
- **Face Recognition**: Siamese networks and one-shot learning
- **Neural Style Transfer**: Artistic image generation using CNNs
- **Object Detection**: Advanced YOLO implementation and R-CNN variants
- **Face Verification**: Binary classification for face recognition systems

### Course 5: Sequence Models
- **Recurrent Neural Networks**: LSTM and GRU implementations
- **Attention Mechanisms**: Multi-head attention and transformer components
- **Character-level RNN**: Text generation and language modeling

## 🏆 ISE Final Test Projects

### 1. Defect Detection Challenge
**Objective**: Classify source code snippets as secure or insecure to identify potential vulnerabilities

#### Key Features:
- **Hybrid BiLSTM Architecture**: Custom neural network combining bidirectional LSTM with attention mechanisms
- **Multi-Model Ensemble**: Integration of multiple models for improved performance
- **Advanced Text Processing**: TF-IDF vectorization and custom feature engineering
- **Performance Optimization**: Achieved ROC score > 0.63 as required

#### Technical Implementation:
- **Models Used**: 
  - Hybrid BiLSTM with attention
  - ResNet-based classifiers
  - Neural network ensembles
- **Data Processing**: 
  - Text preprocessing and tokenization
  - Feature extraction using TF-IDF
  - Stratified cross-validation
- **Evaluation Metrics**: ROC-AUC, precision, recall, F1-score

#### Files Structure:
```
ise-final-test/defect detection/
├── hybrid_bilstm.ipynb          # Main implementation notebook
├── defect_detection.ipynb       # Alternative approaches
├── new.ipynb                    # Experimental models
├── data/                        # Dataset and submissions
├── best_*_model.h5             # Trained model weights
└── final_*_results.pkl         # Performance results
```

### 2. Product Classification Challenge
**Objective**: Classify products using both textual descriptions and visual data

#### Key Features:
- **Multi-Modal Learning**: Combining text and image data for classification
- **Transfer Learning**: Utilizing pre-trained vision models
- **Data Augmentation**: Techniques to improve model generalization
- **Performance Target**: Achieved accuracy > 92% as required

#### Technical Implementation:
- **Architecture**: 
  - Convolutional Neural Networks for image processing
  - Recurrent networks for text analysis
  - Fusion layers for multi-modal integration
- **Data Processing**:
  - Image preprocessing and augmentation
  - Text tokenization and embedding
  - Feature normalization and scaling

#### Files Structure:
```
ise-final-test/product-classification/
├── model.ipynb                  # Main implementation
├── data/                        # Training and test data
└── best_model.h5               # Trained model weights
```

## 🛠️ Technical Stack

### Core Libraries:
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Alternative deep learning implementation
- **Scikit-learn**: Machine learning utilities and preprocessing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn**: Data visualization

### Specialized Tools:
- **NLTK**: Natural language processing
- **OpenCV**: Computer vision operations
- **Transformers**: Pre-trained language models
- **XGBoost/LightGBM**: Gradient boosting algorithms

## 📊 Key Achievements

- **Defect Detection**: Successfully identified security vulnerabilities in code with high accuracy
- **Product Classification**: Achieved multi-modal classification with >92% accuracy
- **Model Optimization**: Implemented ensemble methods and hyperparameter tuning
- **Code Quality**: Clean, well-documented implementations with comprehensive error handling

## 🚀 Getting Started

### Prerequisites:
```bash
pip install -r requirements.txt
```

### Running the Projects:
1. **Defect Detection**: Open `ise-final-test/defect detection/hybrid_bilstm.ipynb`
2. **Product Classification**: Open `ise-final-test/product-classification/model.ipynb`
3. **Coursera Coursework**: Navigate to `dl_coursera_notebooks/` for specific course materials

## 📈 Performance Results

### Defect Detection Challenge:
- **ROC-AUC Score**: >0.63 (Target achieved)
- **Model Architecture**: Hybrid BiLSTM with attention
- **Ensemble Methods**: Weighted voting and stacking

### Product Classification Challenge:
- **Accuracy**: >92% (Target achieved)
- **Multi-Modal Approach**: Text + Image fusion
- **Transfer Learning**: Pre-trained vision models

## 🔬 Research Contributions

- **Hybrid Architectures**: Novel combination of BiLSTM and attention mechanisms
- **Ensemble Methods**: Advanced model combination strategies
- **Multi-Modal Learning**: Effective fusion of textual and visual features
- **Security Applications**: Practical application of ML in cybersecurity

## 📝 Notes

- All models are trained and saved for reproducibility
- Comprehensive documentation and comments throughout the code
- Results and performance metrics are saved for analysis
- Virtual environments are excluded from version control (see `.gitignore`)

## 🤝 Contributing

This repository represents personal learning projects and coursework. Feel free to explore the implementations and adapt them for your own learning or research purposes.

## 📄 License

This project is for educational purposes. Please respect the terms of use for any datasets or course materials referenced.
