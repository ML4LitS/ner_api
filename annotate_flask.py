from flask import Flask, request, jsonify, render_template
import random
import socket
import numpy as np
import json
from optimum.pipelines import pipeline
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig

app = Flask(__name__)

model_path_quantised = '/hps/software/users/literature/textmining/test_pipeline/ml_filter_pipeline/ml_fp_filter/quantised'
model_quantized = ORTModelForTokenClassification.from_pretrained(model_path_quantised,
                                                                 file_name="model_quantized.onnx")
tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised, model_max_length=512, batch_size=4,
                                                    truncation=True)

ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized,
                         aggregation_strategy="first")

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)