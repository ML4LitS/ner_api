"""
Flask App for NER-based Annotations using QEB8L Model

Author: Santosh Tirunagari

This Flask application serves as a platform for performing Named Entity Recognition (NER) annotations using the QEB8L model.
Named Entity Recognition is a natural language processing task that involves identifying entities such as names of persons, organizations,
locations, etc., in text. The QEB8L model is a state-of-the-art NER model known for its accuracy and performance.

Usage:
1. Install the required dependencies by running: `pip install flask`
2. Run this Flask app and access it through a web browser.
3. Paste the text you want to annotate in the provided text area.
4. Click the 'Annotate' button to submit the text to the QEB8L NER model.
5. The annotated entities will be highlighted in the displayed text.

This application demonstrates the integration of the QEB8L NER model with a web interface using Flask, making it user-friendly
and accessible for NER annotation tasks. 

"""

from flask import Flask, request, jsonify, render_template

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


import random
import socket
import numpy as np
import json
from optimum.pipelines import pipeline
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig

import sys
sys.path.append('.')

# from transformers import set_auth_token
#
# # Set the token
# set_auth_token('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')


from entity_linking import retrieve_similar_terms_with_fuzzy_batched
app = Flask(__name__)

model_path_quantised = '/home/stirunag/environments/models/quantised/'
model_quantized = ORTModelForTokenClassification.from_pretrained(model_path_quantised,
                                                                 file_name="model_quantized.onnx")
tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised, model_max_length=512, batch_size=4,
                                                    truncation=True)

ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized,
                         aggregation_strategy="first")

def mapToURL(entity_group, id):
    if not id:
        return "#"
    switcher = {
        'GP': f"https://www.uniprot.org/uniprotkb/{id}/entry",
        'DS': f"http://linkedlifedata.com/resource/umls-concept/{id}",
        'OG': f"http://identifiers.org/taxonomy/{id}",
        'CD': f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={id}"
    }
    return switcher.get(entity_group, "#")


print("Loading.... finished!!!")
def merge_with_same_spans(x_list):
    merged_list = []
    for sublist in x_list:
        if merged_list and merged_list[-1][1] == sublist[0] and merged_list[-1][2] == sublist[2]:
            merged_list[-1][1] = sublist[1]
            merged_list[-1][3] += sublist[3]
        else:
            merged_list.append(sublist)

    return merged_list


@app.route('/')
def index():
    return render_template('index.html', host=socket.gethostbyname(socket.gethostname()))

@app.route('/annotate', methods=['POST'])
def annotate():
    # data = request.get_json()
    # input_text = data['text']
    input_text = request.form.get('text')
    if not input_text:
        return 'No text provided', 400
    # Perform your calculation here. I'm generating random data for demonstration purposes.
    #########################  debugging code###################
    # output = []
    # for i in range(5):
    #     start = random.randint(0, len(input_text))
    #     end = start + random.randint(1, 5)
    #     word = input_text[start:end]
    #     output.append({
    #         'entity_group': random.choice(['OG', 'GP', 'DS', 'CD']),
    #         'score': round(random.random(), 2),
    #         'word': word,
    #         'start': start,
    #         'end': end
    #     })
    # return jsonify(output)
    ###################################################################
    output = ner_quantized(input_text)
    result = [{k: round(float(v), 3) if isinstance(v, np.float32) else v for k, v in res.items()} for res in output]
    # Save input and output to a text file
    with open('annotation_log.txt', 'a') as file:
        json.dump({'Input': input_text, 'Output': result}, file)
        file.write('\n')
        
    return jsonify(result)


@app.route('/annotate_cli', methods=['POST'])
def annotate_cli():
    # data = request.get_json()
    # input_text = data['text']
    input_text = request.form.get('text')
    if not input_text:
        return 'No text provided', 400
    output = ner_quantized(input_text)
    result = [{k: round(float(v), 3) if isinstance(v, np.float32) else v for k, v in res.items()} for res in output]
    # Save input and output to a text file

    x_list_ = []
    for ent in result:
        # print(ent)
        if input_text[int(ent['start']):int(ent['end'])] in ['19', 'COVID', 'COVID-19']:
            ent['entity_group'] = 'DS'
        x_list_.append([ent['start'], ent['end'], ent['entity_group'], input_text[int(ent['start']):int(ent['end'])], ent['score']])

    with open('annotation_cli_log.txt', 'a') as file:
        json.dump({'Input': input_text, 'Output': result}, file)
        file.write('\n')

    return jsonify(x_list_)

@app.route('/annotate_link_cli', methods=['POST'])
def annotate_link_cli():
    input_text = request.form.get('text')
    if not input_text:
        return 'No text provided', 400

    output = ner_quantized(input_text)
    result = [{k: round(float(v), 3) if isinstance(v, np.float32) else v for k, v in res.items()} for res in output]

    if not result:
        return jsonify([])

    x_list_ = []

    # Extracting terms and their corresponding entity groups
    term_entity_pairs = []
    for ent in result:
        if input_text[int(ent['start']):int(ent['end'])] in ['19', 'COVID', 'COVID-19']:
            ent['entity_group'] = 'DS'
        term_entity_pairs.append((input_text[int(ent['start']):int(ent['end'])], ent['entity_group']))

    # Retrieving mapped terms for each entity pair
    mapped_terms_dict = {}
    for term, entity_group in term_entity_pairs:
        mapped_terms = retrieve_similar_terms_with_fuzzy_batched([term], entity_group)
        mapped_terms_dict[term] = mapped_terms[term]

    # Processing results and building output
    for ent in result:
        term = input_text[int(ent['start']):int(ent['end'])]
        if mapped_terms_dict[term][0][1] > 30:
            ent_id = mapped_terms_dict[term][0][2]
            mapped_term = mapped_terms_dict[term][0][0]
            url = mapToURL(ent['entity_group'], ent_id)
            x_list_.append([ent['start'], ent['end'], ent['entity_group'], mapped_term, ent['score'], ent_id, url])
        else:
            url = mapToURL(ent['entity_group'], None)
            x_list_.append([ent['start'], ent['end'], ent['entity_group'], term, ent['score'], None, url])

    # Logging
    with open('annotation_cli_log.txt', 'a') as file:
        json.dump({'Input': input_text, 'Output': x_list_}, file)
        file.write('\n')

    return jsonify(x_list_)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
