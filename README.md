# NER Annotation using QEB8L Model

Author: Santosh Tirunagari

Welcome to the NER Annotation GitHub repository! This repository contains code for a Flask web application that performs Named Entity Recognition (NER) annotations using the QEB8L model.

## Installation

1. First, you need to install the quantised model by following these steps:
   - Visit the [annotation_models](https://github.com/ML4LitS/annotation_models) repository.
   - Follow the instructions provided to download the quantised QEB8L model.

2. Once you have downloaded the quantised model, you can proceed with setting up the Flask application.

## Setup

1. Clone this repository to your local machine:

git clone https://github.com/YourUsername/NER-QEB8L-Annotation.git


2. Install the required dependencies by navigating to the project directory and running:
pip install -r requirements.txt


3. Open the `annotate_flask.py` file in a text editor of your choice.

4. Locate the following line in the code:

model_path_quantised = 'PATH TO THE QUANTISED MODEL'


Replace 'PATH TO THE QUANTISED MODEL' with the actual path where you downloaded the quantised model from the annotation_models repository.

Save the changes to the annotate_flask.py file.

## Usage
1. Make sure you have completed the installation and setup steps.

2. Open a terminal window and navigate to the project directory.

3. Run the Flask application using the following command:

4. python annotate_flask.py

5. Open your web browser and visit http://localhost:5000.

6. In the provided text area, paste the text that you want to annotate.

7. Click the 'Annotate' button to submit the text to the QEB8L NER model.

8. The annotated entities will be highlighted in the displayed text.

## Contributions
Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License
This project is licensed under the CC-by License.
