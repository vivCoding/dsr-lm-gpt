import re
import csv
import string
from typing import List, Set, Tuple
from openai import OpenAI
import scallopy
import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
scallop_root_dir = os.path.abspath(os.path.join(os.path.curdir, "./scl"))

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

id_relation_map = { val: key for key, val in relation_id_map.items()}


default_system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."

# all possible transitive rules
# 20 relationships, transitive involves 3 entites, thus 20 ^ 3 possible transitive rules
all_possible_transitives = [(a, b, c) for a in range(20) for b in range(20) for c in range(20) ]

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
    

def collate_fn(batch):
    queries = [[query] for (_, query, _) in batch]
    sentences = [sentences for (sentences, _, _) in batch]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, _, answer) in batch])
    return (sentences, queries, answers)

def clutrr_loader(batch_size=32, max_entries=10094):
    dataset = CLUTRRDataset("./data/_train.csv")
    # split training and validation date 80/20 rule
    split_idx = int(max_entries * 0.8)
    train_loader = DataLoader([dataset[idx] for idx in range(split_idx)], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader([dataset[idx] for idx in range(split_idx, len(dataset))], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = []
    for idx in range(2, 11):
        ds = CLUTRRDataset(f"./data/_test{idx}.csv")
        for elem in ds:
            test_dataset.append(elem)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader)


class DSRLMModel(torch.nn.Module):
    # TODO organize hyperparams, maybe move to a separate config file or parameterize them
    def __init__(self, gpt_model: str) -> None:
        super(DSRLMModel, self).__init__()

        self.scallop_ctx = scallopy.context.ScallopContext(provenance="difftopbottomkclauses", train_k=3, test_k=3)

        # TODO adjust to appropriate scl file
        self.scallop_ctx.import_file(os.path.join(scallop_root_dir, "bare.scl"))
        # TODO perhaps configure this
        self.scallop_ctx.set_iter_limit(10)
        self.scallop_ctx.set_non_probabilistic(["question"])

        self.reasoner = self.scallop_ctx.forward_function("answer", output_mapping=list(range(len(relation_id_map))))

        self.gpt_model = gpt_model

        # Transitivity probs: Initialize with 0.1
        self.transitivity_probs = torch.tensor(np.ones(len(all_possible_transitives)) / 10, requires_grad=True)
        self.sample_ct = 200

    # returns [ (clean_sentence, set_of_names), ... ]
    def get_contexts(self, sentences: List[str]) -> List[Tuple[str, Set[str]]]:
        i = 0
        contexts = []
        while i < len(sentences):
            sentence = sentences[i]
            names = re.findall("\\[(\w+)\\]", sentence)
            names = set(names)
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
                    names.update(re.findall("\\[(\w+)\\]", sentence))
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
                        names.update(re.findall("\\[(\w+)\\]", sentence))
                        j -= 1
                    break
            contexts.append((final_context, names))
            i += 1

        return contexts

    def prompt_gpt(self, system_prompt: str, prompt: str, logprobs=True, top_logprobs=1, max_tokens=10):
        if logprobs:
            completion = client.chat.completions.create(
                model=self.gpt_model,
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
                model=self.gpt_model,
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": prompt}
                ],
                logprobs=logprobs,
                max_tokens=max_tokens
            )

        # print(completion)
        return completion

    # returns answer and probability of that answeer
    def prompt_for_answer(self, sentences, name1, name2) -> Tuple[str, float]:
        prompt = f"{sentences}\n So {name1} is {name2}'s:"

        completion = self.prompt_gpt(default_system_prompt, prompt)

        answer = completion.choices[0].message.content
        probability = 0
        num = 0

        for logprob in completion.choices[0].logprobs.content:
            probability += np.exp(logprob.logprob)
            num += 1

        # make it lowercase, and remove excess punctuation
        answer = answer.lower().translate(str.maketrans("", "", string.punctuation))
        return answer, probability / num

    # extracts [ (prob, (relation, name1, name2)), ... ]
    def extract_facts(self, sentence_name_pairs):
        facts = defaultdict(int)
        for sentences, names in sentence_name_pairs:
            names = list(names)
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    answer, prob = self.prompt_for_answer(sentences, names[i], names[j])
                    if answer in relation_id_map:
                        facts[(relation_id_map[answer], names[j], names[i])] = max(torch.tensor(prob), facts[(relation_id_map[answer], names[j], names[i])])
                    else:
                        facts[(20, names[j], names[i])] = max(torch.tensor(min(0.5, prob)), facts[(20, names[j], names[i])])
                    answer, prob = self.prompt_for_answer(sentences, names[j], names[i])
                    if answer in relation_id_map:
                        facts[(relation_id_map[answer], names[i], names[j])] = max(torch.tensor(prob), facts[(relation_id_map[answer], names[i], names[j])])
                    else:
                        facts[(20, names[i], names[j])] = max(torch.tensor(min(0.5, prob)), facts[(20, names[i], names[j])])

        listfacts = []
        for key, value in facts.items():
            listfacts.append((value, key))

        return listfacts

    # TODO change to async
    def make_api_calls(sentences, facts):
        pass

    def forward(self, X, phase="train"):
        sentence_batch, queries_batch = X

        contexts_batch = []
        facts_batch = []
        ct = 0
        for sentences in sentence_batch:
            ct += 1
            print("onto ct", ct, end="\r")
            contexts = self.get_contexts(sentences)
            contexts_batch.append(contexts)
            facts = self.extract_facts(contexts)
            facts_batch.append(facts)
        print()

        transitivity_probs = torch.clamp(self.transitivity_probs, 0, 1)

        if phase == 'train':
            sampled_transitive_idx = torch.multinomial(transitivity_probs, self.sample_ct)
        else:
            _, sampled_transitive_idx = torch.topk(transitivity_probs, self.sample_ct)
        probs = transitivity_probs[sampled_transitive_idx]

        transitives = [all_possible_transitives[i] for i in sampled_transitive_idx]
        transitive_relations = [[(prob, relation) for prob, relation in zip(probs, transitives)] for _ in range(len(sentence_batch))]

        result = self.reasoner(
            question=queries_batch,
            context=facts_batch,
            transitive=transitive_relations)

        return result
    

class Trainer:
    def __init__(self) -> None:
        self.train_loader, self.val_loader, self.test_loader = clutrr_loader(batch_size=32, max_entries=500)
        self.model = DSRLMModel(gpt_model=os.getenv("GPT_MODEL"))
        self.optimizer = torch.optim.Adam([self.model.transitivity_probs], lr=0.01)

    def accuracy(self, y_pred, y):
        batch_size = len(y)
        pred = torch.argmax(y_pred, dim=1)
        num_correct = len([() for i, j in zip(pred, y) if i == j])
        return (num_correct, batch_size)

    def loss(self, y_pred: torch.Tensor, y: torch.Tensor):
        _, dim = y_pred.shape
        # y is a 1d array, where each elem contains the idx of the relation
        # convert it into a 2d array to compute loss
        ground_truths = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
        return torch.nn.functional.binary_cross_entropy(y_pred, ground_truths)

    def train(self, num_epochs: int):
        for i in range(1, num_epochs + 1):
            self.train_epoch(i)
            self.test_epoch(i)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_count = 0
        total_correct = 0
        total_loss = 0
        iterator = tqdm(self.train_loader)
        for (i, batch) in enumerate(iterator):
            self.optimizer.zero_grad()
            sentences, queries, y = batch
            y_pred = self.model((sentences, queries), phase="train")
            loss = self.loss(y_pred, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            (num_correct, batch_size) = self.accuracy(y_pred, y)
            total_count += batch_size
            total_correct += num_correct
            correct_perc = 100. * total_correct / total_count
            avg_loss = total_loss / (i + 1)

            iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")
    
    def test_epoch(self, epoch: int):
        self.model.eval()
        total_count = 0
        total_correct = 0
        total_loss = 0
        with torch.no_grad():
            iterator = tqdm(self.test_loader)
            for (i, batch) in enumerate(iterator):
                sentences, queries, y = batch
                y_pred = self.model((sentences, queries), 'test')
                print("we loss", i)
                loss = self.loss(y_pred, y)
                total_loss += loss.item()

                (num_correct, batch_size) = self.accuracy(y_pred, y)
                total_count += batch_size
                total_correct += num_correct
                correct_perc = 100. * total_correct / total_count
                avg_loss = total_loss / (i + 1)

                iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")
        # Save model
        if total_correct / total_count > self.max_accu:
            self.max_accu = total_correct / total_count
            torch.save(self.model, f"./trans.best.model")
        torch.save(self.model, f"./trans.latest.model")

    def get_rules(self, threshold=0.6):
        pred_probs = self.model.transitivity_probs.reshape(-1)
        rules = pred_probs > threshold
        indices = rules.nonzero()
        rules = sorted([(pred_probs[index].item(), [id_relation_map[e] for e in all_possible_transitives[index]]) for index in indices], reverse=True)
        return rules

def pretty_print_rules(rules):
    rule_str = '\n'.join([f"{p};{r}" for p, r in rules])
    print(rule_str)
    return rule_str

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(num_epochs=3)
    
    rules = trainer.get_rules()
    pp_rules = pretty_print_rules(rules)
    with open("learned_rules.txt", "w") as f:
        f.write(pp_rules)



