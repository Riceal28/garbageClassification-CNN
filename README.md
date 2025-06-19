# Garbage Classification CNN

This project uses a convolutional neural network (CNN) to implement the task of classifying garbage images, capable of identifying 10 different types of garbage: batteries, biowaste, cardboard, clothing, glass, metal, paper, plastic, shoes, and regular garbage.

## Dataset

The project uses a garbage image dataset containing 10 categories:

1. Battery

2. Biological

3. Cardboard

4. Clothes

5. Glass

6. Metal

7. Paper

8. Plastic

9. Shoes

10. Trash

The dataset is automatically divided according to the following proportions:

- Training set: 70%

- Validation set: 20%

- Test set: 10%

## Model architecture

The CNN model architecture is as follows:

```python
model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3), kernel_regularizer=l2(0.0001)),
MaxPooling2D(2,2),
Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.0001)),
MaxPooling2D(2,2),
Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.0001)),
GlobalAveragePooling2D(),
Dense(128, activation='relu'),
Dropout(0.3),
Dense(64, activation='relu'),
Dropout(0.3),
Dense(10, activation='softmax', dtype='float32')
])
```

## Training configuration

- **Image size**: 128×128 pixels
- **Batch size**: 64
- **Training epochs**: 500 (using early stopping mechanism)
- **Optimizer**: Adam (Learning rate = 0.001)
- **Loss function**: Categorical cross entropy
- **Data enhancement**:
- Random left and right flip
- Random brightness adjustment
- Random contrast adjustment
- Random saturation adjustment
- Random hue adjustment

## How to run

1. **Install dependent libraries**:
```bash
pip install -r requirements.txt
```
If you are in conda environment, please use the following command
```bash
conda env create -f environment.yml
```
2. **Prepare dataset**:
- Dataset download link (you can also download directly from the project): https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data
- Download the dataset and unzip it to the `garbage-classification` folder in the project directory
- The dataset structure should be as follows:
```
garbage-classification/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/
```

3. **Run Jupyter Notebook**:
```bash
jupyter notebook garbage_classification_cnn.ipynb
```

4. **Execute code**:
- Execute all cells in the order in the Notebook
- The best model (`best_model.h5`) is automatically saved during training

## Dependencies

Main dependent libraries:
- TensorFlow 2.10
- Keras
- NumPy
- Matplotlib
- OpenCV

For a complete list of dependencies, see `requirements.txt`

## Results

The model showed good learning ability during training:
- Training accuracy continued to improve
- Verification accuracy increased steadily
- Use mixed precision training to accelerate the training process
- Early stopping mechanism prevents overfitting (patience = 10 cycles)

After training, the model can be evaluated on the test set to obtain the final classification accuracy.

## Notes

1. Make sure GPU is available to accelerate training (mixed precision training is enabled in the code)

2. Dataset partitioning code is included (automatically partitioning training set/validation set/test set)

3. The best model will be saved during training
