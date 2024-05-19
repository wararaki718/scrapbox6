from pydantic import BaseModel


class LlamaConfig(BaseModel):
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct"
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model"
    max_seq_len: int = 128
    max_batch_size: int = 4
