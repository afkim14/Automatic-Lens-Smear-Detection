# EECS 395 Assignment #1
### Automatic Lens Smear Detection
### Andres Kim, YaNing Wang, Stacy Montgomery

## Directory Structure
```
|   img_process.py
├── test (test directory to test functionality of img_process.py)
├── pre_processed_images (development code)
│   └── bins
|       └── *.jpg (intermediate pre-processed images)
└── processed_output (distribution code)
    └── bins
        └── finalMask.jpg (Mask of Detected Smears)
```

## Setup
This script is run with Python 3. Please make sure you have dependencies installed.
```
pip3 install opencv-python
pip3 install pillow
```

## Running the Script
Make sure you have a directory compromised of images for smear detection. Then run the following:
```
python3 img_process.py [image_dir]
```
For example, we have also included a test directory for usage testing.
```
python3 img_process.py test/
```

The script will output two directories: `pre_processed_images/` and `processed_output`. `pre_processed_images` will contain the modified images using Histogram Equalization, Gaussian Blur, and more. These images will be split into bins for better processing and results. `processed_output` will contain the masked .jpg images of the possible smears based on the pre-processed images. There will be a masked output for each bin. 
