# Toxic Comment Detector

Toxic Comment Detector is a project designed to identify and classify toxic comments across categories like toxic, severe toxic, obscene, threat, insult, and identity hate. Built with DistilBERT, this model achieves a strong mean AUC of 0.9797 on validation data.

## About the Project

This repository contains a machine learning model trained on the Jigsaw Toxic Comment Classification Challenge dataset to detect various forms of toxicity in text. The project includes a user-friendly web application powered by Gradio, allowing anyone to test the model with ease. You can explore the live demo or access the model directly on Hugging Face.

## What's Included

- `app.py`: A Gradio-based web app for real-time toxicity analysis.
- `toxic_detector.py`: Handles model training and inference logic.
- `train.csv`: The Jigsaw dataset used to train the model.
- `requirements.txt`: Lists all dependencies needed to run the project.
- `.gitattributes` and `.gitignore`: Git configuration files for smooth version control.

## Getting Started

To try it locally:
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the web app: `python app.py`.

## Try It Out

- **Live Demo**: Test the model interactively on [Hugging Face Spaces](https://huggingface.co/spaces/thequantumcoder/toxic-detector).
- **Model Access**: Explore the trained model on [Hugging Face](https://huggingface.co/thequantumcoder/toxic-detector).

## How It Performs

The model was trained over 3 epochs using DistilBERT with a learning rate of 2e-5 and a batch size of 32. It delivers impressive validation AUC scores:
- Toxic: 0.9721
- Severe Toxic: 0.9787
- Obscene: 0.9910
- Threat: 0.9519
- Insult: 0.9858
- Identity Hate: 0.9987

## License

This project is licensed under the MIT License.
