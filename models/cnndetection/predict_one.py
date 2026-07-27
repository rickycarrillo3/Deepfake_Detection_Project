"""
Score a single image with the CNNDetection model from the terminal.

Usage:
    cnndetection_env/bin/python predict_one.py /path/to/image.png
"""

import argparse

import model


def main():
    parser = argparse.ArgumentParser(description="Run CNNDetection predict() on one image.")
    parser.add_argument("image_path", help="Path to the image to score.")
    args = parser.parse_args()

    model.load()
    score = model.predict(args.image_path)

    print(f"{args.image_path}: {score:.4f} (fake-probability)")


if __name__ == "__main__":
    main()
