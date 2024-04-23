"""code from https://www.datacamp.com/tutorial/fine-tuning-llama-2"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def load_merge_model(base_model, new_model):
    # Reload model in FP16 and merge it with LoRA weights
    load_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    model = PeftModel.from_pretrained(load_model, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def train(model, tokenizer, dataset):
    # Fine-tune model
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_porj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()

    # Save model
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

def evaluate(model, tokenizer):
    # Eval model
    model.eval()
    # Define pipeline
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        return_full_text=True, 
        max_new_tokens=512, 
        temperature=0.1, 
        repetition_penalty=1.1,
        )

    prompt = "Who is Leonardo Da Vinci?"
    result = pipe(f"{prompt}", eos_token_id=tokenizer.eos_token_id)
    print("Result: {0}".format(result[0]['generated_text']))


if __name__ == "__main__":
    # Model from Hugging Face hub
    base_model = "NousResearch/Llama-2-7b-chat-hf"

    # New instruction dataset
    guanaco_dataset = "mlabonne/guanaco-llama2-1k"

    # Fine-tuned model
    new_model = "llama-2-7b-chat-guanaco"

    # Load dataset
    dataset = load_dataset(guanaco_dataset, split="train")

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train(model, tokenizer)

    model, tokenizer = load_merge_model(base_model, new_model)

    evaluate(model, tokenizer)
