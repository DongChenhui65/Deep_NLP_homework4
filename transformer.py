
import torch
import jieba
from transformers import  BertTokenizer,GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from datasets import Dataset


def preprocesss(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def load_dataset(corpus, tokenizer ):
    encoded = tokenizer(corpus, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = encoded["input_ids"].clone()
    dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "labels": labels.tolist()
    })
    return dataset



def finetune_model(model, train_dataset, output_dir, tokenizer):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )


    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()


# 加载gpt2
model = GPT2LMHeadModel.from_pretrained( "uer/gpt2-distil-chinese-cluecorpussmall").to(torch.device('cuda'))
gpt2_tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
gpt2_embedding_weights = model.transformer.wte.weight
vocab = gpt2_tokenizer.get_vocab()
corpus = './/data//corpus_sentence.txt'
chinese_corpus = preprocesss(corpus)
# 加载数据集
dataset = load_dataset(chinese_corpus, gpt2_tokenizer)
finetune_model(model, dataset, "./gpt2-finetuned", gpt2_tokenizer)
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(torch.device('cuda'))

    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for p in punctuation:
        generated_text = generated_text.replace(p * 3, p)

    generated_text = ''.join(generated_text.split())
    return generated_text


prompt = "马背上伏的是个高瘦的汉子，汉子手里拿了一把长剑，剑长三尺，"

text_gpt2 = generate_text(model, gpt2_tokenizer, prompt)
print(f"GPT-2: {text_gpt2}")