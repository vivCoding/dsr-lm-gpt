import re
import csv
from openai import OpenAI
import scallopy
import os
import torch
from dotenv import load_dotenv
from collections import defaultdict
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

default_system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."


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


class DSRLMModel:
    # TODO organize hyperparams, maybe move to a separate config file or parameterize them
    def __init__(self) -> None:
        self.scallop_ctx = scallopy.context.ScallopContext(provenance="difftopbottomkclauses", train_k=3, test_k=3)

        # TODO adjust to appropriate scl file
        self.scallop_ctx.import_file(os.path.join(scallop_root_dir, "manual_rules.scl"))
        self.scallop_ctx.set_non_probabilistic(["question"])
        # TODO perhaps configure this
        self.scallop_ctx.set_iter_limit(10)

        self.reasoner = self.scallop_ctx.forward_function("answer", output_mapping=list(range(len(relation_id_map))))

    def get_contexts(self, sentences):
        i = 0
        contexts = []
        while i < len(sentences):
            sentence = sentences[i]
            names = re.findall("\\[(\w+)\\]", sentence)
            clean_sentence = sentence.replace("[", "").replace("]", "") + "."
            final_context = clean_sentence
            if clean_sentence[-1] != ".":
                final_context += "."
            j = i + 1
            while len(names) < 2:
                if j < len(sentences):
                    sentence = sentences[j]
                    clean_sentence = sentence.replace("[", "").replace("]", "") + "."
                    final_context += " " + clean_sentence
                    if clean_sentence[-1] != ".":
                        final_context += "."
                    new_names = re.findall("\\[(\w+)\\]", sentence)
                    for name in new_names:
                        if name not in names:
                            names.append(name)
                    j += 1
                else:
                    j = i - 1
                    while len(names) < 2:
                        sentence = sentences[j]
                        clean_sentence = sentence.replace("[", "").replace("]", "") + "."
                        if clean_sentence[-1] != ".":
                            final_context = clean_sentence + ". " + final_context
                        else:
                            final_context = clean_sentence + " " + final_context
                        new_names = re.findall("\\[(\w+)\\]", sentence)
                        for name in new_names:
                            if name not in names:
                                names.append(name)
                        j -= 1
                    break
            contexts.append((final_context, names))
            i += 1

        return contexts

    def prompt_gpt(self, system_prompt, prompt, logprobs=False, top_logprobs=0, max_tokens=10):
        if logprobs:
            completion = client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:personal::9E5UHiJr",
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": prompt}
                ],
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens
            )
        else:
            completion = client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:personal::9E5UHiJr",
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": prompt}
                ],
                logprobs=logprobs,
                max_tokens=max_tokens
            )

        return completion

    def prompt_for_answer(self, sentences, name1, name2):
        prompt = f"{sentences}\n So {name1} is {name2}'s:"

        completion = self.prompt_gpt(default_system_prompt, prompt)

        answer = completion.choices[0].message.content

        return answer
    def extract_facts(self, sentence_name_pairs):
        facts = defaultdict(int)
        for sentences, names in sentence_name_pairs:
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    answer = self.prompt_for_answer(sentences, names[i], names[j])
                    if answer in relation_id_map:
                        facts[(relation_id_map[answer], names[j], names[i])] = max(torch.tensor(0.99), facts[(relation_id_map[answer], names[j], names[i])])

                    answer = self.prompt_for_answer(sentences, names[j], names[i])
                    if answer in relation_id_map:
                        facts[(relation_id_map[answer], names[i], names[j])] = max(torch.tensor(0.99), facts[(relation_id_map[answer], names[i], names[j])])

        listfacts = []
        for key, value in facts.items():
            listfacts.append((value, key))

        return listfacts

    def scallop(self, query, facts):
        result = self.reasoner(question=query, context=facts)
        return result

    def forward(self, X):
        sentences, query, answer = X
        contexts = self.get_contexts(sentences)
        facts = self.extract_facts(contexts)
        query = [[query]]
        facts = [facts]
        result = self.scallop(query, facts)

        return result


if __name__ == "__main__":
    dataset = CLUTRRDataset("./data/_test2.csv")
    scallop_root_dir = os.path.abspath(os.path.join(os.path.curdir, "./scl"))
    model = DSRLMModel()
    preds = []
    for example_num in range(2):
        output = model.forward(dataset[example_num])
        output = output.squeeze()
        argmax = torch.argmax(output)
        pred = ""
        for rel, id in relation_id_map.items():
            if argmax.item() == id:
                pred = rel
        preds.append(pred)

    print(preds)

