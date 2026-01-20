# quick_check_classes.py
import os

AUDIO_DIR = r'C:\Users\Jordan\Desktop\Birds'
class_names = sorted(os.listdir(AUDIO_DIR))
print("Your model knows these species:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")