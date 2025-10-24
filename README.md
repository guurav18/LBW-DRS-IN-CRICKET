# LBW-Detection-DRS

A Python-based Decision Review System (DRS) simulator focused on **Leg Before Wicket (LBW) detection** in cricket. 
This project tracks the ball in video footage, predicts its trajectory, and automatically decides whether the batsman is **OUT** or **NOT OUT** based on pitch, impact, and height analysis.

## Features
- Ball tracking using OpenCV (color detection + contours)
- Trajectory prediction using polynomial regression
- LBW decision logic based on pitch, impact, and height
- Text-to-speech announcement of the decision
- Visualized pitch boundaries for better analysis

## Technologies Used
- Python 3
- OpenCV
- NumPy
- pyttsx3 (Text-to-Speech)

## How to Run
1. Install dependencies:
```bash
pip install opencv-python numpy pyttsx3
