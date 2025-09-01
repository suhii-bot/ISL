# ISL: Indian Sign Language to Text Translator

A real-time Indian Sign Language (ISL) to Text Translator using Python, MediaPipe, and Convolutional Neural Networks (CNN). This project recognizes hand gestures via webcam and converts them into text, helping bridge communication between speech/hearing-impaired individuals and others. It is scalable, accurate, and designed for accessibility.

## Features

- **Real-time ISL Recognition:** Detects and translates Indian Sign Language gestures live from webcam input.
- **Gesture-to-Text Conversion:** Converts recognized hand signs directly into readable text.
- **Modular Model Design:** Easy to extend for more gestures or languages.
- **Accuracy & Speed:** Uses a trained CNN for high accuracy and rapid prediction.
- **Accessibility Focus:** Designed for ease of use by individuals with speech or hearing impairments.
- **Media Capture:** Supports uploading snap videos or images for recognition and documentation.

## Tech Stack

- **Python:** Main programming language.
- **MediaPipe:** For robust hand landmark detection and gesture tracking.
- **OpenCV:** For video stream capture and image processing.
- **TensorFlow/Keras:** For building and training the CNN gesture classification model.
- **Jupyter Notebook / Colab:** For model training and experimentation.
- **NumPy & Pandas:** Data manipulation and preprocessing.
- **Other Libraries:** Matplotlib for visualization, tqdm for progress bars.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suhii-bot/ISL.git
   cd ISL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main application:**
   ```bash
   python main.py
   ```
   Show ISL gestures in front of your webcam. The recognized text will be displayed on the screen.

4. **Uploading Snap Videos or Images:**
   - Place your video (`.mp4`, `.avi`) or image (`.jpg`, `.png`) files in the `media/` folder.
   - Use the provided scripts (e.g., `process_media.py`) to analyze these files and extract gesture predictions.

## Project Structure

- `main.py` - Main entry point for webcam-based recognition.
- `model/` - Contains CNN model architecture and weights.
- `utils/` - Utility functions for data processing, visualization, etc.
- `media/` - Folder for storing snap videos and images for gesture analysis.
- `docs/` - Documentation and example usage.
- `requirements.txt` - List of dependencies.
- `README.md` - Project overview and instructions.


## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add (such as new gestures, improved models, or additional media).

## License

This project is currently unlicensed(Educational purpose only) 

## About

Developed and maintained by [suhii-bot](https://github.com/suhii-bot).
