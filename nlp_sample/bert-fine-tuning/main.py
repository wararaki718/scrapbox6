from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from metrics import MetricsCalculator
from text import TextTokenizer


def main() -> None:
    dataset_name = "yelp_review_full"
    dataset = load_dataset(dataset_name)
    print(f"load data: {len(dataset['train'])}")

    model_name = "google-bert/bert-base-cased"
    tokenizer = TextTokenizer(model_name)
    print("tokenizer loaded.")

    tokenized_datasets = dataset.map(tokenizer.tokenize, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    print(f"train dataset: {len(small_train_dataset)}")

    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    print(f"test dataset: {len(small_eval_dataset)}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    print(f"model loaded.")

    # define hyper-parameters
    training_args = TrainingArguments(output_dir="test_trainer")
    metrics_calculator = MetricsCalculator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=metrics_calculator.compute
    )
    print("trainer defined.")

    trainer.train()

    print("DONE")


if __name__ == "__main__":
    main()
