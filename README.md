# Generating Declarative Statements from QA Pairs

There are already some rule-based models that can accomplish this task, but I haven't seen any transformer-based models that can do so. Therefore, I trained this model based on `Bart-base` to transform QA pairs into declarative statements.

I compared the my model with other rule base models, including 

> [paper1](https://aclanthology.org/D19-5401.pdf) (2019), which proposes **2 Encoder Pointer-Gen model**

and

> [paper2](https://arxiv.org/pdf/2112.03849.pdf) (2021), which propose RBV2 **model**

**Here are results compared to 2 Encoder Pointer-Gen model (on testset released by paper1)**

Test on testset

| Model   | 2 Encoder Pointer-Gen(2019) | BART-base  |
| ------- | --------------------------- | ---------- |
| BLEU    | 74.05                       | **78.878** |
| ROUGE-1 | 91.24                       | **91.937** |
| ROUGE-2 | 81.91                       | **82.177** |
| ROUGE-L | 86.25                       | **87.172** |

Test on NewsQA testset

| Model   | 2 Encoder Pointer-Gen | BART       |
| ------- | --------------------- | ---------- |
| BLEU    | 73.29                 | **74.966** |
| ROUGE-1 | **95.38**             | 89.328     |
| ROUGE-2 | **87.18**             | 78.538     |
| ROUGE-L | **93.65**             | 87.583     |

Test on free_base testset

| Model   | 2 Encoder Pointer-Gen | BART       |
| ------- | --------------------- | ---------- |
| BLEU    | 75.41                 | **76.082** |
| ROUGE-1 | **93.46**             | 92.693     |
| ROUGE-2 | **82.29**             | 81.216     |
| ROUGE-L | **87.5**              | 86.834     |



**As paper2 doesn't release its own dataset, it's hard to make a fair comparison. But according to results in paper2, the Bleu and ROUGE score of their model is lower than that of MPG, which is exactly the 2 Encoder Pointer-Gen model.**

| Model        | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------ | ---- | ------- | ------- | ------- |
| RBV2         | 74.8 | 95.3    | 83.1    | 90.3    |
| RBV2+BERT    | 71.5 | 93.9    | 82.4    | 89.5    |
| RBV2+RoBERTa | 72.1 | 94      | 83.1    | 89.8    |
| RBV2+XLNET   | 71.2 | 93.6    | 82.3    | 89.4    |
| MPG          | 75.8 | 94.4    | 87.4    | 91.6    |

There are reasons to believe that my model performs better than RBV2.

To sum up,my model performs nearly as well as the SOTA rule-based model evaluated with BLEU and ROUGE score. However the sentence pattern is lack of diversity.

(It's worth mentioning that even though I tried my best to conduct objective tests, the testsets I could find were more or less different from what they introduced in the paper.)



## Requirement

```bash
conda create -n QA2D python=3.7
conda activate QA2D
pip install transformers
pip install datasets # if you want to train your own model
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```



## How to use

The model is uploaded to Huggingface, which means that you can use universal api to load the model. For example,

```python
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained("MarkS/QA2D")
model = BartForConditionalGeneration.from_pretrained("MarkS/QA2D")

input_text = "question: what day is it today? answer: Tuesday"
input = tokenizer(input_text, return_tensors='pt')
output = model.generate(input.input_ids)
result = tokenizer.batch_decode(output, skip_special_tokens=True)
print(result)
```



If you just want to transform your QA-pairs to declarative sentences, you can do as follow:

#### Step 1

Transform the input QA pairs into the following form:

"question: \<question text\> answer: \<answer text\>

For example:

> question: What is the purpose of this passage ? answer: To present a research result.

Then put them into "QA2D/input/input.txt", one question per line.

#### Step 2

Run the following command

```python
python transform.py
```

The output will be in "QA2D/output/output.txt"

For example:

> the purpose of this passage is to present a research result



You may also use a [GEC model](https://github.com/SoyMark/gector_roberta) to make the declarative statements more fluent, though this may bring about some minor changes to the statements, rendering them a bit inconsistent with the original questions. It happens especially when declarative answers contain uncommon expressions.



#### Update(2022.4.16)

Constrained decode method is available in the new release of transformers (4.17.0 or later release). I've done some test on it. It truly makes sure the force words are in the generated sentences. However, the overall performance declines. Since QA2D is a relatively easy task, key words missing rarely occurs. So whether to use it depends on your goal.

Here is an example of using constrained decode method.

```python
input_text = "question: what day is it today? answer: Tuesday"

input = tokenizer(input_text, return_tensors='pt')
force_words = [' '+'Tuesday'] # note that the space character is vital
force_words = tokenizer(force_words, add_special_tokens=False)
output = model.generate(
        input.input_ids,
        force_words_ids=force_words.input_ids,
        num_beams=50, # big beam size is beneficial
        remove_invalid_values=True
						)
result = tokenizer.batch_decode(output, skip_special_tokens=True)
print(result)
```

