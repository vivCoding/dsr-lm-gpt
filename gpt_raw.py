import json
import re
import csv
import string
from typing import List, Tuple
from openai import OpenAI
import scallopy
import os
import torch
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
import time
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
default_system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: " \
                "daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, " \
                "grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."

class CLUTRRDataset:
    def __init__(self, file_path):
        self.data = [instance for instance in list(csv.reader(open(file_path)))[1:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sentences = [s.strip() for s in self.data[i][2].split(".") if s.strip() != ""]

        query = eval(self.data[i][3])
        query = (query[0], query[1])

        answer = self.data[i][5]
        return sentences, query, answer


def prompt_gpt(sentences: List[str], query: Tuple[str]):
    prompt = "".join(sentences).replace("[", "").replace("]", "") + ".\n"
    name1, name2 = query
    prompt += f"So {name2} is {name1}'s:"
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": default_system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        logprobs=True,
        top_logprobs=1,
        max_tokens=10
    )
    answer = completion.choices[0].message.content
    # make it lowercase, and remove excess punctuation
    return answer.lower().translate(str.maketrans("", "", string.punctuation))


if __name__ == "__main__":
    start_time = time.time()
    results_root_dir = "./results/gpt_raw/"
    
    test_datasets = [f"_test{idx}" for idx in range(2, 11)]
    accuracies = []
    for test_name in test_datasets:
        dataset = CLUTRRDataset(f"./data/{test_name}.csv")
        scallop_root_dir = os.path.abspath(os.path.join(os.path.curdir, "./scl"))
        gpt_model = os.getenv("GPT_MODEL")

        preds = []
        actual = []

        file = open(os.path.join(results_root_dir, f"result{test_name}_{gpt_model}.txt"), "w")
        for example_num in range(len(dataset)):
            sentences, query, answer = dataset[example_num]
            pred = prompt_gpt(sentences, query)
            preds.append(pred)
            actual.append(answer)
            file.write(f"{pred}, {answer}\n")
            print(f"Completed: {example_num + 1} / {len(dataset)}", end="\r")
        file.close()
        print()
        
        correct = 0
        for idx in range(len(preds)):
            if preds[idx] == actual[idx]:
                correct += 1
        accuracy = correct / len(preds)
        accuracies.append({ test_name: accuracy })

        print(f"Accuracy {test_name}:", accuracy)
    
    with open(os.path.join(results_root_dir, "gpt_raw_accs.json"), "w") as f:
        json.dump(accuracies, f, indent=4)
    print("Total secs:", time.time() - start_time)
    
