from datasets import load_dataset
import os
import argparse
import torch
import logging
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
#from transformers.models.bart.modeling_bart import shift_tokens_right


logger = logging.getLogger(__name__)

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

class BART_for_answer_extension:
    def __init__(self, model_path, data_path=""):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        if(data_path != ""):
            self.data = load_dataset(data_path)
        else:
            self.data = load_dataset('csv', data_files={'train': "./data/train.csv", 'validation': "./data/dev.csv"})
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.train_dataset = self.data['train']
        self.valid_dataset = self.data['validation']

    #data processing

    def generate_input(self, one_question):
        ret = dict()
        ret['input_text'] = "question: %s answer: %s" % (one_question['question'], one_question['answer'])
        ret['label_text'] = "%s" % one_question['turker_answer']
        return ret


    def data_encoder(self, example_batch):
        input_encodings = self.tokenizer.batch_encode_plus(example_batch['input_text'], padding='max_length', max_length=96, truncation=True)
        label_encodings = self.tokenizer.batch_encode_plus(example_batch['label_text'], padding='max_length', max_length=64, truncation=True)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': label_encodings['input_ids']
        }
        return encodings
    
    def data_preparation(self):
        self.train_dataset = self.train_dataset.map(self.generate_input)
        self.train_dataset = self.train_dataset.map(self.data_encoder, batched=True)

        self.valid_dataset = self.valid_dataset.map(self.generate_input)
        self.valid_dataset = self.valid_dataset.map(self.data_encoder, batched=True)

from typing import Dict, List

class MyDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch]) 
        labels = torch.stack([torch.tensor(example['labels']) for example in batch])
        decoder_input_ids = shift_tokens_right(labels, 1) # model.config.pad_token_id is 1
        attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
        labels[labels[:, :] == 1] = -100 
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids, 
            'labels': labels,
        }

model_dir = './model'

def main():
    Bart_model = BART_for_answer_extension("facebook/bart-base")
    #Bart_model.model.cuda() 
    Bart_model.data_preparation()
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps", 
        eval_steps=500,
        learning_rate=1e-5,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        do_train=True,
        warmup_steps=300,
        weight_decay=0.01,
        logging_dir=model_dir+'/logs',
        overwrite_output_dir=True,
        save_strategy='epoch',
        logging_first_step=True,
        logging_steps=20,
    )

    trainer = Trainer(
        model=Bart_model.model,
        args=training_args,
        train_dataset=Bart_model.train_dataset,
        eval_dataset=Bart_model.valid_dataset,
        data_collator=MyDataCollator()
    )
    trainer.train()
    trainer.save_model()
    results = {}

    logger.info("*** Evaluate ***")
    eval_output = trainer.evaluate()
    output_eval_file = os.path.join(model_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(eval_output.keys()):
            logger.info("  %s = %s", key, str(eval_output[key]))
            writer.write("%s = %s\n" % (key, str(eval_output[key])))
    results.update(eval_output)
    return results
'''
    parser = argparse.ArgumentParser(description="fine tune a model")
    #args = parser.parse_args() # to do
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                    help="the model to be initiated with")
    parser.add_arguement("--num_train_epoch, default=1, type=int,
                    help="number of training epoch")
    parser.add_argument("--eval_per_epoch", default=1, type=int,
                    help="How many times it evaluates on dev set per epoch and saves a checkpoint")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints will be written")
    parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size to train a model")
'''
if __name__ == '__main__':
    main()
