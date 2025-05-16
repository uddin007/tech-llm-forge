# tech-llm-forge

## **Fine-Tuning Mistral-7B for Databricks Technical Documentation**  

## **Introduction**  
Large Language Models (LLMs) like Mistral-7B are powerful out-of-the-box, but fine-tuning them for domain-specific tasks can dramatically improve their performance. In this repo, we’ll explore how we fine-tuned **Mistral-7B** to answer technical questions:  

1. **Dataset Preparation** – Structuring Q&A pairs for fine-tuning  
2. **Model Training** – Using Parameter-Efficient Fine-Tuning (PEFT) with LoRA  
3. **Evaluation** – Measuring improvements with perplexity, BLEU, and ROUGE scores  

---

## **Why Fine-Tune Mistral-7B?**  
Mistral-7B is a powerful open-weight LLM, but it lacks specialized knowledge about, say, **Databricks SQL functions** (e.g., `stack()`). Fine-tuning adapts the model to:  
- **Generate accurate technical answers**  
- **Include proper SQL or PySpark syntax**  
- **Avoid hallucinations in responses**  

### **Key Technologies Used**  
- **Hugging Face Transformers** – For model loading and training  
- **PEFT (LoRA)** – Efficient fine-tuning with low-rank adaptation  
- **4-bit Quantization** – Reduces GPU memory usage (A100 in Google Colab)
- **Perplexity** - Lower perplexity means the model is much more "confident" (less surprised) when generating responses
- **BLEU & ROUGE Metrics** – For further evaluating response quality  

---

## **Dataset Preparation**  
We used a **custom Q&A dataset** (5,000 examples) with:  
- **Questions**: *"How does the `stack` function work in Databricks SQL?"*  
- **Answers**: Detailed explanations with correct SQL examples.
The original data is prepared and make available by Databricks at `Databricks Demos` and llm/databricks-documentation repository. It is a labeled training set (questions+answers) with state-of-the-art technical answers from Databricks support team. This .parquet format dataset is downloaded as .csv and used to fine tune our model. 

### **Dataset Formatting**  
Each training sample was structured as:  
```json
{
  "instruction": "What is the stack function in Databricks SQL?",
  "input": "",
  "output": "The stack function generates multiple rows...",
  "context": "Databricks SQL documentation"
}
```
This format helps the model learn **instruction-following behavior**.  

---

## **Fine-Tuning with LoRA**  
Instead of full fine-tuning (which is expensive), we used **LoRA (Low-Rank Adaptation)**, which trains only small adapter layers while keeping the base model frozen.  

### **Key Training Parameters**  
```python
peft_config = LoraConfig(
    r=16,           # Rank of the adaptation matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,       # Mixed-precision training
    logging_steps=10,
    save_strategy="epoch"
)
```
In **LoRA Configuration Parameters**, we used:
* r=16: This defines the rank of the low-rank decomposition matrices. Lower values (like 4 or 8) use fewer parameters but may limit adaptation capacity, while higher values (like 16 or 32) increase capacity but use more parameters. Our choice of 16 strikes a good balance for technical documentation.
* lora_alpha=32: This is a scaling factor that affects the magnitude of the LoRA updates. The effective learning rate is approximately lora_alpha/r (32/16 = 2) times the base learning rate. Higher values can lead to stronger adaptation but risk instability.
* target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]: These are the attention mechanism components we're fine-tuning:
* q_proj: Query projection matrices
* k_proj: Key projection matrices
* v_proj: Value projection matrices
* o_proj: Output projection matrices
This targets the entire attention mechanism of the transformer, which is effective for adapting to new domains like technical documentation.
lora_dropout=0.05: A small dropout applied during training to prevent overfitting in the LoRA layers.
bias="none": No bias parameters are trained.
task_type="CAUSAL_LM": Specifies that you're fine-tuning for causal language modeling (predicting the next token).

These four projection matrices are where our model learns what information to prioritize when generating text. By fine-tuning these components, we achieve:
* Domain Adaptation: we're teaching the model which parts of technical documentation are most relevant to each other
* Pattern Recognition: The model learns technical documentation-specific patterns and relationships
* Efficient Parameter Update: These matrices contain a significant portion of the model's "knowledge" while representing only a fraction of its total parameters

When fine-tuning on Databricks technical documentation, modifying these attention components allows the model to:
* Better understand technical terminology relationships
* Maintain consistency across longer technical explanations
* Learn proper formatting and structure of technical documentation
* Recognize domain-specific connections that weren't as important in the model's general training

In **Training Arguments**, we used:
* per_device_train_batch_size=4: Small batch size (4 examples per device), suitable for single GPU and memory constraints.
* gradient_accumulation_steps=4: Accumulates gradients over 4 steps before updating weights, effectively creating a virtual batch size of 16 (4×4). This helps achieve the benefits of larger batch sizes without exceeding memory limits.
* learning_rate=2e-4: A moderate learning rate (0.0002) appropriate for this fine-tuning. 
* num_train_epochs=3: The model will see the entire dataset 3 times. We used this common value for fine-tuning considering sufficient data.
* fp16=True: Uses 16-bit floating point precision (half precision) to reduce memory usage and speed up training.
* logging_steps=10: Records training metrics every 10 steps for monitoring progress.
* save_strategy="epoch": Saves a checkpoint at the end of each epoch.

This configuration is well-suited for fine-tuning an LLM on technical documentation, striking a good balance between adaptation capacity, training efficiency, and avoiding overfitting.
- **4-bit Quantization** was applied to fit Mistral-7B into Colab’s A100 GPU.  

---

## **Evaluating Model Performance**  
### **Perplexity (PPL) – Measuring Confidence**  
- **Base Model PPL**: **27.18**  
- **Fine-Tuned PPL**: **11.12** (**59.1% improvement**)  
Lower perplexity means the model is **more certain** about its answers.  

### **ROUGE Scores – Measuring Answer Quality**  
| Metric  | Base Model | Fine-Tuned | Improvement |  
|---------|------------|------------|-------------|  
| ROUGE-1 | 0.3402     | **0.3880** | +14%        |  
| ROUGE-2 | 0.1110     | **0.2022** | +82%        |  
| ROUGE-L | 0.1878     | **0.2535** | +35%        |  

- **ROUGE-2 (Bigram Overlap) improved the most**, meaning the model learned **better technical phrasing**.  
- **BLEU decreased slightly (0.0655 → 0.0567)**, which is expected since BLEU penalizes paraphrasing—even if correct.  

### **Qualitative Improvements**  
- **Before Fine-Tuning**:  
  *"The stack function reverses two arguments."* (Incorrect)  
- **After Fine-Tuning**:  
  *"The `stack(n, expr1, expr2, ...)` function generates `n` rows. Example:  
  ```sql  
  SELECT stack(2, 1, 2, 3) AS (col1, col2);  
  ```"*  

---

![Unknown](https://github.com/user-attachments/assets/64792415-93e7-4a30-a2a3-28737ba4e215)

---

## **Conclusion**  
Fine-tuning Mistral-7B with **LoRA and 4-bit quantization** significantly improved its ability to answer Databricks related technical questions accurately. The **59% perplexity drop and 82% ROUGE-2 improvement** prove that even small adapters can make LLMs domain experts.  
