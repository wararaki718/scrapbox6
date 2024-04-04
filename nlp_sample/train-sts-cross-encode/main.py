import logging
import math
from pathlib import Path

from sentence_transformers import CrossEncoder, LoggingHandler, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader

from loader import Loader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    sts_dataset_path = Path("datasets/stsbenchmark.tsv.gz")
    if not sts_dataset_path.exists():
        util.http_get(
            "https://sbert.net/datasets/stsbenchmark.tsv.gz",
            str(sts_dataset_path),
        )
        logger.info("data downloaded.")

    model_name = "distilroberta-base"
    model = CrossEncoder(model_name, num_labels=1)

    logger.info("training")
    train_samples, valid_samples, test_samples = Loader.load(sts_dataset_path)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=64)

    evaluator = CECorrelationEvaluator.from_input_examples(valid_samples, name="sts-dev")

    num_epochs = 4
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logger.info(f"Warmup-steps: {warmup_steps}")

    model_save_path = Path("output/training_stsbenchmark")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )

    logger.info("evaluate")
    model = CrossEncoder(model_save_path)
    evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name="sts-test")
    evaluator(model)
    logger.info("DONE")


if __name__ == "__main__":
    main()
