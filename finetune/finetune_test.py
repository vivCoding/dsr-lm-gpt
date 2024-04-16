from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::9E5UHiJr",
    messages=[
        {"role": "system", "content": "Given a sentence and the names of two people choose the relationship between the people from the following options: daughter, sister, son, aunt, father, husband, granddaughter, brother, nephew, mother, uncle, grandfather, wife, grandmother, niece, grandson, son-in-law, father-in-law, daughter-in-law, mother-in-law. Answer in one word."},
        {"role": "user", "content": "Dorothy's brother Michael and her went to get ice cream.\n So Dorothy is Michael's:"}
    ],
    logprobs=True,
    top_logprobs=20,
    max_tokens=20
)

print(completion)

print(completion.choices[0].message)
