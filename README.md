# Generating Declarative Statements from QA Pairs

There are already some rule-based models that can accomplish this task, but I haven't seen any transformer-based models that can do so. Therefore, I trained this model based on `Bart-base` to transform QA pairs into declarative statements.

I compared the my model with other rule base models, including 

> https://aclanthology.org/D19-5401.pdf (2019)

and

> https://arxiv.org/pdf/2112.03849.pdf (2021)

The result is that this model perform nearly as well as the rule-based model evaluated with BLEU and ROUGE score. However the sentence pattern is lack of diversity.

To be honest, even though I tried my best to conduct objective tests, it's just a rough comparison, because the trainset and testset I could find were more or less different from what they used in the paper.



## Requirement

```bash
conda create -n QA2D python=3.7
conda activate QA2D
pip install transformers
pip install datasets
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then, download the model and unzip it to "QA2D/model".

> https://drive.google.com/file/d/1wrKNMDd0n1KPDb1kl7jG4kgSSTY44kEs/view?usp=sharing



## How to use

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