# test_bird.py
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import soundfile as sf


# ===== 1. DEFINE YOUR MODEL (same as training) =====
class BirdNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(BirdNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = torch.nn.Linear(128 * 5 * 13, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)

        self.dropout = torch.nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ===== 2. LOAD TRAINED MODEL =====
def load_model(model_path, class_names):
    num_classes = len(class_names)
    model = BirdNet(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model


# ===== 3. PROCESS AUDIO FILE =====
def process_audio(audio_path, sample_rate=22050, n_mels=64, duration=3):
    """Process audio file into mel-spectrogram (same as training)"""

    # Load audio
    waveform, sr = sf.read(audio_path)

    # Convert to torch tensor
    if len(waveform.shape) == 1:
        waveform = torch.FloatTensor(waveform).unsqueeze(0)
    else:
        waveform = torch.FloatTensor(waveform.T)

    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Trim or pad to fixed duration
    target_length = sample_rate * duration
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

    # Create mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=1024
    )
    mel_spec = mel_transform(waveform)

    # Convert to dB
    mel_spec = T.AmplitudeToDB()(mel_spec)

    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0)

    return mel_spec


# ===== 4. PREDICT BIRD SPECIES =====
def predict_bird(model, audio_path, class_names):
    """Predict bird species from audio file"""

    # Process audio
    spectrogram = process_audio(audio_path)

    # Make prediction
    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get result
    predicted_species = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100

    return predicted_species, confidence_pct, probabilities[0]


# ===== 5. MAIN FUNCTION =====
if __name__ == '__main__':
    # ===== CONFIGURATION =====
    MODEL_PATH = 'bird_classifier.pth'
    #AUDIO_FILE = r'C:\Users\Jordan\Desktop\Birds\American Robin\13602-0.wav'  # ← PUT YOUR TEST FILE HERE
    AUDIO_FILE = r'C:\Users\Jordan\Downloads\Robin.mp3'
    # Your class names (MUST match training order!)
    CLASS_NAMES = [
        'American Robin',
        'Bewicks Wren',
        'Northern Cardinal',
        'Northern Mockingbird',
        'Song Sparrow',
        # ... add all your species here in alphabetical order
    ]

    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, CLASS_NAMES)

    # Make prediction
    print(f"\nAnalyzing audio file: {AUDIO_FILE}")
    species, confidence, all_probs = predict_bird(model, AUDIO_FILE, CLASS_NAMES)



    # Print results
    print(f"\n{'=' * 50}")
    print(f"PREDICTION: {species}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print(f"{'=' * 50}")

    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top3_probs, top3_indices = torch.topk(all_probs, 3)
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices), 1):
        print(f"  {i}. {CLASS_NAMES[idx]}: {prob.item() * 100:.2f}%")



