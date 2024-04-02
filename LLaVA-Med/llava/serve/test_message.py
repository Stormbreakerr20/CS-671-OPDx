import argparse
import json
import requests
import base64
from llava.conversation import default_conversation
import time

def generate_response(worker_addr, model_name, prompt, max_new_tokens,image_url):
    headers = {"User-Agent": "LLaVA Client"}
    pload = {
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "stop": default_conversation.sep,
        "img_url": image_url,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True)
    output = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(default_conversation.sep)[-1]
    return output

def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        if len(models) == 0: print("Model not found")

        # print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        # print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], args.message)
    prompt = conv.get_prompt()

    print(prompt.replace(conv.sep, "\n"), end="")

    response = generate_response(worker_addr, args.model_name, prompt, args.max_new_tokens,args.image_url)
    time.sleep(10)
    with open("t.txt", "w") as f:
        f.write(prompt.replace(conv.sep, "\n"))
        f.write(response)

    print()
    print(response, end="")
    print()
    
if __name__ == "__main__":
    while(True):
        parser = argparse.ArgumentParser()
        parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
        parser.add_argument("--worker-address", type=str)
        parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
        parser.add_argument("--max-new-tokens", type=int, default=10000)

        # Custom Input
        print()
        inp1 = input("Enter prompt: ")
        print()
        inp2 = input("Enter a publically accessible image url: ")

        if inp2 != "": inp1 = inp1 + " \n<image>"
        parser.add_argument("--message", type=str, default=inp1) 
        parser.add_argument("--image_url", type=str, default=inp2) 
        # what is in the image \n<image>
        args = parser.parse_args()
        main()
        time.sleep(5)
