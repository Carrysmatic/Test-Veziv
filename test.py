from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from datasets import load_dataset

model_name = "llama3"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('json', data_files={'train': './test.json'})

def tokenize_function(examples):
    return tokenizer(examples['prompt'], examples['response'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

prompt = "Clientul solicita dezvoltarea unei aplicatii de tip Glovo. Specifica urmatoarele sectiuni:,
 Sistem de logistică a rider-ului.,
 Înregistrarea restaurantelor în aplicație.,
 Primirea comenzilor de către clienți.,
 Posibilitatea de a adăuga sau elimina produse din meniul restaurantului.,
 Plata cu cardul."
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(input_ids, max_length=500)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)