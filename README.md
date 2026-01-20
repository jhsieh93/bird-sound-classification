# Bird Sound Classification using Deep Learning

A convolutional neural network (CNN) model for automated classification of bird species from audio recordings. The model achieves 84% validation accuracy in distinguishing between five common North American bird species.

## Problem Statement

Automated bird species identification from audio has applications in wildlife monitoring, ecological research, and citizen science initiatives. This project demonstrates end-to-end development of a production-ready audio classification pipeline.

## Species Classification

The model classifies audio recordings into five species:
- American Robin
- Bewick's Wren
- Northern Cardinal
- Northern Mockingbird
- Song Sparrow

## Technical Approach

**Data Source:** Audio recordings from [Xeno-canto](https://xeno-canto.org/), a community database of bird sounds

**Audio Processing Pipeline:**
- Conversion of raw audio to mel-spectrograms for visual representation of frequency content
- Data augmentation techniques to improve model generalization
- Regularization to prevent overfitting

**Model Architecture:**
- Convolutional Neural Network (CNN) optimized for audio classification
- PyTorch implementation for training and inference
- Model achieves 84% validation accuracy

## Project Structure

```
AudioClassifier/
├── main.py              # Model training pipeline
├── Test_bird.py         # Inference and evaluation
├── Audio Organizer.py   # Data preprocessing and organization
├── bird_classifier.pth  # Trained model weights
└── requirements.txt     # Python dependencies
```

## Key Learning Outcomes

- Audio signal processing and spectrogram generation
- CNN architecture design for audio classification tasks
- Overfitting detection and mitigation strategies
- Model training, validation, and evaluation workflows
- Production deployment considerations for ML models

## Technologies Used

- **Framework:** PyTorch
- **Audio Processing:** librosa, torchaudio
- **Data Science:** NumPy, pandas
- **Visualization:** matplotlib

## Results

- **Validation Accuracy:** 84%
- Successfully implemented data augmentation and regularization techniques
- Model demonstrates strong generalization to unseen audio samples

## Future Enhancements

- Expand to additional bird species
- Implement real-time classification pipeline
- Deploy as web API for mobile applications
- Integration with edge devices for field deployment

## Author

Jordan Hsieh | [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/jhsieh93)

Technical Program Manager with background in digital signal processing and AI/ML, transitioning into AI product development.
