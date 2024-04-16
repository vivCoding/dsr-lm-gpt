import json
import re
import csv
import random

system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."

relation_opposite = {
    'daughter': ["mother", "father"],
  'sister': ["brother"],
  'son': ["mother", "father"],
  'aunt': ["niece", "nephew"],
  'father': ["son", "daughter"],
  'husband': ["wife"],
  'granddaughter': ["grandmother", "grandfather"],
  'brother': ["sister"],
  'nephew': ["aunt", 'uncle'],
  'mother': ["son", "daughter"],
  'uncle': ["niece", "nephew"],
  'grandfather': ["granddaughter", "grandson"],
  'wife': ["husband"],
  'grandmother': ["granddaughter", "grandson"],
  'niece': ["aunt", "uncle"],
  'grandson': ["grandmother", "grandfather"],
  'son-in-law': ["mother-in-law", "father-in-law"],
  'father-in-law': ["daughter-in-law", "son-in-law"],
  'daughter-in-law': ["mother-in-law", "father-in-law"],
  'mother-in-law': ["daughter-in-law", "son-in-law"]
}

class CLUTRRDataset:
    def __init__(self, file_path):
        self.data = [instance for instance in list(csv.reader(open(file_path)))[1:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sentences = [s.strip() for s in self.data[i][2].split(".") if s.strip() != ""]

        query = eval(self.data[i][3])
        query = (query[0], query[1])

        proof_state = self.data[i][8]
        proof_state = proof_state.replace(":", ",").replace("{", "").replace("}", "")
        index = proof_state.find("[", proof_state.find("[") + 1)
        proof_state = proof_state[:index] + proof_state[index+1:]
        index = proof_state.find("]", proof_state.find("]") + 1)
        proof_state = proof_state[:index] + proof_state[index+1:]
        proof_state = eval(proof_state)
        return sentences, query, proof_state


if __name__ == "__main__":
    dataset = CLUTRRDataset("../data/_train.csv")
    num_examples = 100
    file_path = "train_finetune.jsonl"
    open(file_path, "w")
    json_file = open(file_path, "a")
    count = 0
    for i in range(len(dataset)):
        sentences, query, proof_state = dataset[i]
        for sentence in sentences:
            names = re.findall("\\[(\w+)\\]", sentence)
            if len(names) != 2:
                continue
            relationship = ""
            firstFirst = True
            for state in proof_state:
                if type(state) != tuple:
                    continue
                if state[0] == names[0] and state[2] == names[1]:
                    relationship = state[1]
                    break
                elif state[2] == names[0] and state[0] == names[1]:
                    relationship = state[1]
                    firstFirst = False
                    break
            if relationship == "":
                continue
            clean_sentence = sentence.replace("[", "").replace("]", "") + "."
            prompt = ""
            if firstFirst:
                prompt = f"{clean_sentence}\n So {names[1]} is {names[0]}'s:"
            else:
                prompt = f"{clean_sentence}\n So {names[0]} is {names[1]}'s:"
            if not clean_sentence.find(relationship):
                found = False
                for relation in relation_opposite[relationship]:
                    if clean_sentence.find(relation):
                        found = True
                if not found:
                    continue
            item = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}, {"role": "assistant", "content": relationship}]}
            json.dump(item, json_file)
            json_file.write("\n")
            count += 1

