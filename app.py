import gradio as gr
from PIL import Image
from predict import predict
def run(img):
    if isinstance(img, dict): img = Image.fromarray(img['composite'])
    return predict(img)
gr.Interface(run, gr.Sketchpad(type='numpy'), gr.Label(num_top_classes=3), title='Распознавание цифр').launch()

import requests
MODEL = 'qwen2.5:0.5b'
def ask(prompt, system='', temperature=0.7):
    r = requests.post('http://localhost:11434/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']
if __name__ == '__main__':
    print(ask('Объясни одним коротким предложением, почему рукописную цифру {digit} можно легко перепутать с другой цифрой или почему она так выглядит.'))
    