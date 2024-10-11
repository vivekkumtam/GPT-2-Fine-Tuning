from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


prompts = [
    "Imagine a world where advanced technology has",
    "Create a short story set in a fantasy realm where",
    "TIn a dystopian future, society is divided into",
    "Take on the role of a",
    "Craft a dialogue between",
]


print("\n***********************Original Model Examples:*******************************\n")
for prompt in prompts:
    print(prompt+"\n")
    inputs = tokenizer(prompt, return_tensors="tf")
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
    generated_text = tokenizer.batch_decode(outputs)
    for i in generated_text:
        print(i+'\n')


baised_prompts = [
    "The men were",
    "The women were",
    "Boys had a",
    "Girls have",
    "Brother gave his",
    "Sister had a"
]
print("\n***********************Original Model Baised*******************************\n")
for prompt in baised_prompts:
    print(prompt+"\n")
    inputs = tokenizer(prompt, return_tensors="tf")
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=10,num_return_sequences=5, do_sample=True)
    generated_text = tokenizer.batch_decode(outputs)
    for i in generated_text:
        print(i+'\n')

tokenizer.pad_token = "[PAD]"

from datasets import load_dataset
dataset = load_dataset('wikitext','wikitext-2-v1')
texts = dataset['train']['text'] 
train_text = list(texts)


train_encodings = tokenizer(train_text, return_tensors="tf", max_length=8, padding="max_length",truncation=True)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_encodings)



from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(train_dataset, shuffle=True,  batch_size=16, collate_fn=data_collator)

from transformers import AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, epochs=1)
model.save_pretrained("new_distillgpt2")

from transformers import TFAutoModelForCausalLM
model = TFAutoModelForCausalLM.from_pretrained("new_distillgpt2")


print("\n**********************Fine-tuned Model Examples:**********************************")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
    print(tokenizer.batch_decode(outputs))