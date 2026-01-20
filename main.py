import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import os



# ===== 1. DATASET CLASS =====
class BirdSoundDataset(Dataset):
    """Loads bird audio files and converts to spectrograms"""
# Here is where my data is, these are my settings
    def __init__(self, audio_dir, sample_rate=22050, n_mels=64, duration=3):
        """
        audio_dir: folder with subfolders for each bird species
        Example structure:
            audio_dir/
                robin/
                    robin1.wav
                    robin2.wav
                sparrow/
                    sparrow1.wav
                    sparrow2.wav
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels

        # Collect all audio files and labels
        self.audio_files = []
        self.labels = []
        self.class_names = sorted(os.listdir(audio_dir))

        for label_idx, bird_species in enumerate(self.class_names):
            species_dir = os.path.join(audio_dir, bird_species)
            if os.path.isdir(species_dir):
                for audio_file in os.listdir(species_dir):
                    if audio_file.endswith(('.wav', '.mp3')):
                        self.audio_files.append(os.path.join(species_dir, audio_file))
                        self.labels.append(label_idx)

        # Create mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024
        )
# This is how many audio files I have
    def __len__(self):
        return len(self.audio_files)
# Here's how to load, process, and return these files.
    def __getitem__(self, idx):
        import soundfile as sf

        # Load audio using soundfile instead of torchaudio
        waveform, sr = sf.read(self.audio_files[idx])

        # Convert to torch tensor and ensure correct shape
        if len(waveform.shape) == 1:
            # Mono audio
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
        else:
            # Stereo audio - transpose to (channels, samples)
            waveform = torch.FloatTensor(waveform.T)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Trim or pad to fixed duration
        target_length = self.sample_rate * self.duration
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (dB)
        mel_spec = T.AmplitudeToDB()(mel_spec)

        return mel_spec, self.labels[idx]


# ===== 2. NETWORK ARCHITECTURE =====
class BirdNet(nn.Module):
    def __init__(self, num_classes):
        super(BirdNet, self).__init__()
        # Input: (batch, 1, 64, ~130) - mel bins × time frames
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Calculate flattened size after convs and pools
        # This will depend on your spectrogram dimensions
        # You may need to adjust this based on actual output
        self.fc1 = nn.Linear(128 * 5 * 13, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Conv block 3
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ===== 3. TRAINING FUNCTION =====
def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-4)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for spectrograms, labels in train_loader:
            # ===== MOVE DATA TO GPU =====
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            # ============================
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        train_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                # ===== MOVE DATA TO GPU =====
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                # ============================
                outputs = model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')

    return model


# ===== 4. MAIN EXECUTION =====
if __name__ == '__main__':
    # Setup
    AUDIO_DIR = r'C:\Users\Jordan\PycharmProjects\AudioClassifier\Birds'  # Your audio folder path
    BATCH_SIZE = 16
    NUM_EPOCHS = 20

    # ===== ADD GPU SUPPORT =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # ============================

    # Load dataset
    print("Loading dataset...")
    dataset = BirdSoundDataset(AUDIO_DIR)
    print(f"Found {len(dataset)} audio files")
    print(f"Classes: {dataset.class_names}")

    # ... data loading code ...

    # Create model
    num_classes = len(dataset.class_names)
    model = BirdNet(num_classes=num_classes)
    model = model.to(device)  # ← MOVE MODEL TO GPU

    print(f"\nModel architecture:")
    print(model)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


    # Train
    print(f"\nStarting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, device=device)

    # Save
    torch.save(model.state_dict(), 'bird_classifier.pth')
    print("\nModel saved as 'bird_classifier.pth'")