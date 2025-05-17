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
Instead of full fine-tuning (which is expensive), we used **LoRA (Low-Rank Adaptation)**, which trains only small adapter layers while keeping the base model frozen. LoRA is a parameter-efficient fine-tuning technique that dramatically reduces the computational and memory costs of adapting large language models like Mistral to specific domains. Rather than updating all model parameters (which could be billions for models like Mistral), LoRA works by:
* Keeping the pre-trained model weights frozen (unchanged)
* Injecting small, trainable "adapter" matrices into specific layers of the model
* Using low-rank decomposition to minimize the number of trainable parameters

For any weight matrix W in the original model, LoRA decomposes the weight update ΔW as:
**ΔW = A × B**
where:
A is a matrix of shape (d × r)
B is a matrix of shape (r × k)
r is the "rank" parameter (which you set to 16)
d and k are the original matrix dimensions

This creates a low-rank approximation of the weight update, which requires training only (d × r) + (r × k) parameters, instead of (d × k) parameters.

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

```python
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

### **Parameter Calculation for LoRA Fine-tuning of Mistral-7B**
Mistral-7B has these key specifications: 7B total parameters, Hidden dimension size of 4,096, 32 attention heads and Head dimension of 128 (4,096 ÷ 32). For each of 4-Attention Projection Matrices in Mistral-7B: Input dimension (d) = 4,096
and Output dimension (k) = 4,096. For each targeted matrix, we calculate: Number of parameters = (d × r) + (r × k) where r = 16 (your LoRA rank). For one projection matrix: Number of parameters = (4,096 × 16) + (16 × 4,096) = 65,536 + 65,536 = 131,072. Since, Mistral-7B has 32 transformer layers, in each layer, we're targeting 4 projection matrices, so, total LoRA parameters: 32 layers × 4 matrices × 131,072 parameters = 16,777,216 parameters. Considering, full fine-tuning would require updating all 7 billion parameters, with LoRA, we're only training about 16.8M parameters, which is: 16.8M ÷ 7B = 0.0024 = 0.24% of the full model parameters.

**4-bit quantization memory footprint**
Additionally, with 4-bit quantization, base model size becomes: 7B parameters × 0.5 bytes/parameter = ~3.5GB (compared to ~14GB in fp16 or ~28GB in fp32). However, we kept LoRA parameters still in higher precision, bfloat16 in our case, since they're the only trainable components. That means, total LoRA parameters: 16.8M (as calculated previously) and memory for LoRA in bfloat16: 16.8M × 2 bytes = 33.6MB. So, the total Memory Requirements would be ~3.5GB for the quantized base model (frozen), ~33.6MB for the trainable LoRA parameters with additonal overhead for Optimizer states (using 8-bit paged_adamw_8bit optimizer), Activations, Gradients (only for LoRA parameters) and Batch data. Despite loading the model in 4-bit, our configuration has:
* bnb_4bit_compute_dtype=torch.bfloat16: Calculations are done in bfloat16
* fp16=True: Mixed precision training
To ensure that the model parameters are stored in 4-bit but temporarily converted to bfloat16 during computation, maintaining reasonable numerical precision. This will allow Mistral-7B to fit on consumer GPUs with 8-12GB VRAM while maintaining most of the model quality using just ~25% of the memory.

| Approach              | Parameters              | Memory (approx.) | 
|-----------------------|-------------------------|------------------|
| Full FP32 fine-tuning | 7B                      |28GB              | 
| Full FP16 fine-tuning | 7B                      | 14GB             | 
| 4-bit base + LoRA     | 16.8M trainable (0.24%) | 3.5GB + 33.6MB   | 

---

## **Evaluating Model Performance**  
### **Perplexity (PPL) – Measuring Confidence**  
- **Base Model PPL**: **27.18**  
- **Fine-Tuned PPL**: **11.12** (**59.1% improvement**)  
Perplexity measures how "surprised" a language model is by the test data. Mathematically, it's the exponentiated average negative log-likelihood of a sequence. Here, Lower values indicate better performance, say, if PPL = 10, roughly interpreted as "the model is as confused as if it had to choose uniformly among 10 options for each word". It Signifies:
* Domain adaptation indicator: A significant drop in perplexity after fine-tuning suggests your model has successfully adapted to the technical documentation domain
* Fluency measure: Lower perplexity generally correlates with more fluent text generation
* Prediction confidence: As you noted, lower perplexity means your model is more confident when generating technical content

### **ROUGE Scores – Measuring Answer Quality**  
| Metric  | Base Model | Fine-Tuned | Improvement |  
|---------|------------|------------|-------------|  
| ROUGE-1 | 0.3402     | **0.3880** | +14%        |  
| ROUGE-2 | 0.1110     | **0.2022** | +82%        |  
| ROUGE-L | 0.1878     | **0.2535** | +35%        |  

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
A set of metrics that measure overlap between generated text and reference text, with more emphasis on recall:
- ROUGE-N: Measures n-gram overlap (ROUGE-1 for unigrams, ROUGE-2 for bigrams)
- ROUGE-L: Measures longest common subsequence, capturing in-sequence matches
- ROUGE-S: Measures skip-bigram overlap, allowing for gaps between matched words
It signifies:
- Content coverage: Higher ROUGE suggests our model captures more of the important information
- Completeness measure: Particularly important for technical documentation where missing details could be problematic
- Sequence preservation: ROUGE-L helps evaluate if your model maintains proper ordering of technical steps/explanations
In this case:
- ROUGE-1: +14% Improvement shows better adaptation to the technical vocabulary of Databricks technical documentation
- ROUGE-2: +82% Improvement outlines fine-tuned model is much better at capturing technical phrases and domain-specific expressions that commonly appear together in Databricks documentation
- ROUGE-L: +35% Improvement indicates the fine-tuned model produces text with a sequence and structure much closer to original documentation

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
