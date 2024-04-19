# Facial Paralysis Detection

This repository hosts the code for detecting facial paralysis from video inputs using machine learning techniques. The project analyzes video data to identify symptoms of facial paralysis effectively.

- **Video Processing**: Processes video data to extract facial keypoints using `mmpose_video.py`.
- **Dataset Management**: Manages datasets specific to facial paralysis with `fp_dataset.py`.
- **Model Training**: Trains detection models in `learning.py`.
- **Main Execution**: Orchestrates the workflow through `main.py`.

## Usage

Run the main script:
```bash
python main.py
```

Explore the demonstration notebook:
```bash
jupyter notebook demo.ipynb
```

## License

Licensed under the MIT License. See LICENSE.md for details.

## Author

Ali Mottaghi (mottaghi@stanford.edu)
