import os
import csv
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
import scallopy
import time
from dotenv import load_dotenv
load_dotenv()

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

system_prompt = "Given a sentence and the names of two people choose the relationship between the people from the following options: " \
                "daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, " \
                "grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."

scallop_root_dir = os.path.abspath(os.path.join(os.path.curdir, "./scl"))

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

class TestModel:
    # TODO organize hyperparams, maybe move to a separate config file or parameterize them
    def __init__(self) -> None:
        self.scallop_ctx = scallopy.context.ScallopContext(provenance="difftopbottomkclauses", train_k=3, test_k=3)
        
        # TODO adjust to appropriate scl file
        self.scallop_ctx.import_file(os.path.join(scallop_root_dir, "manual_rules.scl"))
        self.scallop_ctx.set_non_probabilistic(["question"])
        # TODO perhaps configure this
        self.scallop_ctx.set_iter_limit(10)

        self.reasoner = self.scallop_ctx.forward_function("answer", output_mapping=list(range(len(relation_id_map))))
    
    def forward(self):
        # 2d arrays, representing list of rules/facts in each problem
        # refer to the scl files to see the types
        # len(questions) = len(contexts) = batch_size
        questions = [
            [ ("donald", "dorothy") ],
            [ ("alice", "bob") ]
        ]
        contexts = [
            [
                # for each name-pair relation, there's an associated probability
                # scallop doesn't require you to put in all 20 probabilities
                # neither does it require that they add to 1
                # doesn't really matter though since we're kinda doing it anyways
                (torch.tensor(0.85), (2, "donald", "michael")),
                (torch.tensor(0.95), (4, "donald", "michael")),
                (torch.tensor(0.15), (0, "michael", "dorothy")),
                (torch.tensor(0.85), (1, "michael", "dorothy")),
            ],
            [
                (torch.tensor(0.05), (2, "alice", "carmen")),
                (torch.tensor(0.95), (4, "alice", "carmen")),
                (torch.tensor(0.15), (0, "carmen", "bob")),
                (torch.tensor(0.85), (1, "carmen", "bob")),
            ]
        ]
        # returns tensor shape (batch_size, len(relation_id_map))
        result = self.reasoner(question=questions, context=contexts)
        return result

        
if __name__ == "__main__":
    start_time = time.time()

    # TODO add model and train/test loops
    model = TestModel()
    pred = model.forward()
    print (pred)

    print("Total secs:", time.time() - start_time)
    print("ok")

