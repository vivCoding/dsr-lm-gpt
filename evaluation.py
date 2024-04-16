import json
import re
import csv
from openai import OpenAI
import random

relation_id_map = {
  'daughter': 0,
  'sister': 1,
  'son': 2,
  'aunt': 3,
  'father': 4,
  'husband': 5,
  'granddaughter': 6,
  'brother': 7,
  'nephew': 8,
  'mother': 9,
  'uncle': 10,
  'grandfather': 11,
  'wife': 12,
  'grandmother': 13,
  'niece': 14,
  'grandson': 15,
  'son-in-law': 16,
  'father-in-law': 17,
  'daughter-in-law': 18,
  'mother-in-law': 19,
  'nothing': 20,
}

system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."


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



def scallop():
    pass

def get_contexts(sentences):
    i = 0
    contexts = []
    while i < len(sentences):
        sentence = sentences[i]
        names = re.findall("\\[(\w+)\\]", sentence)
        clean_sentence = sentence.replace("[", "").replace("]", "") + "."
        final_context = clean_sentence + "."
        j = i
        while len(names) < 2:
            sentence = sentences[j]
            clean_sentence = sentence.replace("[", "").replace("]", "") + "."
            final_context += clean_sentence + "."
            names += re.findall("\\[(\w+)\\]", sentence)
            j += 1
        contexts.append((final_context, names))
        i += 1

    return contexts
def extract_facts(sentence_name_pairs):
    facts = []
    for sentences, names in sentence_name_pairs:
        for i in range(len(names)):
            for j in range(i, len(names)):
                prompt = f"{sentences}.\n So {names[i]} is {names[j]}'s:"
                print(prompt)

                # completion = client.chat.completions.create(
                #     model="ft:gpt-3.5-turbo-0125:personal::9E5UHiJr",
                #     messages=[
                #         {"role": "system",
                #          "content": "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."},
                #         {"role": "user",
                #          "content": prompt}
                #     ],
                #     logprobs=True,
                #     top_logprobs=20,
                #     max_tokens=20
                # )
                #
                # answer = completion.choices[0].message
                #
                # facts.append((1, (answer, names[i], names[j])))
    return facts

def scallop(facts):
    updated_facts = []
    return updated_facts
def evaluate(test_data):
    dataset = CLUTRRDataset(test_data)
    for sentences, query, answer in dataset:
        contexts = get_contexts(sentences)
        facts = extract_facts(contexts)
        updated_facts = scallop(facts)
        print(updated_facts)


if __name__ == "__main__":
    dataset = CLUTRRDataset("../data/_train.csv")
    num_examples = 100
    file_path = "train_finetune.jsonl"
    open(file_path, "w")

