#!/usr/bin/env python3

import os
from glob import glob
from gpt4all import GPT4All


model_path = os.path.join(os.getcwd(), 'weights')
prompt = "What is the capital of France?"
max_tokens=1024

os.makedirs(model_path, exist_ok=True)

for filename in glob(os.path.join(model_path, '*.gguf')):
    print(f"\nChecking {filename}, asking:\n{prompt}\n")

    model = GPT4All(filename,
                    model_path=model_path, 
                    device='gpu')
    with model.chat_session():
        response = model.generate(prompt, max_tokens=max_tokens)
        print(response)
