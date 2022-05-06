from transformers import BartTokenizer, BartForConditionalGeneration
import torch

tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
device = torch.device("cuda")

def get_response(model, input_text, max_length=64, cuda=True):
    input = tokenizer(input_text, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
    input = input.to(device)
    output = model.generate(input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            max_length=max_length,
                            num_beams=5,
                            )
    #torch.cuda.empty_cache()
    result = tokenizer.batch_decode(output, skip_special_tokens=True)
    return result

def main(model_path, input_file_path, output_file_path,  batch_size=16):
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.cuda()
        with open(output_file_path, mode="w") as writer:
            with open(input_file_path, mode="r")as reader:
                inputs = []
                one_sentence = reader.readline()
                cnt = 0
                while(one_sentence!=""):
                    cnt += 1
                    inputs.append(one_sentence)
                    if (cnt % batch_size == batch_size - 1):
                        outputs = get_response(model, inputs)
                        for j in range(0, len(outputs)):
                            writer.write("%s\n" % outputs[j])
                        inputs = []  # clear inputs
                    one_sentence = reader.readline()
                if(len(inputs) != 0): #handle the last (incomplete) batch
                    outputs = get_response(model, inputs)
                    for j in range(0, len(outputs)):
                        writer.write("%s\n" % outputs[j])
        writer.close()

if __name__ == "__main__":
    best_model_path = "MarkS/bart-base-qa2d"
    input_file = "./input/input.txt"
    output_file = "./output/output.txt"
    main(best_model_path, input_file, output_file, batch_size=16)
