import ssl
import argparse
from loguru import logger
import cahp.config as config
from cahp.config import init_logger
from cahp.utils.data_utils import load_model, load_dataset
from cahp.pruning.pruner import Pruner

# MAC users / SSL handling
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    """Main function to parse arguments and initiate the CAHP pruning process."""
    init_logger()
    args = parse_arguments()
    config.verbose = args.verbose
    
    try:
        model_entity = load_model(args.model_path)
        logger.info(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        logger.error(f"Critical Error: Failed to load HuggingFace model. {type(e).__name__}: {e}")
        logger.info("CAHP currently only supports HuggingFace Transformer models. Goodbye!")
        sys.exit(1)

    # Load dataset splits (expects a directory containing 'train' and 'test' folders)
    train_data, test_data = load_dataset(args.dataset_path)

    # Initialize the Orchestrator
    pipeline = Pruner(
        model=model_entity,
        train_data=train_data,
        test_data=test_data,
        retrain_epochs=args.retrain_epochs
    )

    # Execute the CAHP pipeline
    pipeline.prune(
        pool_size=args.pool_size,
        seed=args.seed,
        poly_deg=args.poly_deg
    )


def parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments for the CAHP pruning algorithm."""
    parser = argparse.ArgumentParser(description='CAHP: Complementary Attention Head Pruning')

    # Path Arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the HuggingFace model directory (containing config.json).')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory containing train/test splits.')

    # CAHP Hyperparameters
    parser.add_argument('--pool_size', type=int, default=32,
                        help='Spatial resolution (B) for attention map pooling. Defaults to 32.')
    parser.add_argument('--poly_deg', type=int, default=6,
                        help='Polynomial degree for the MSS knee-finding algorithm. Defaults to 6.')
    
    # Training & Execution
    parser.add_argument('--retrain_epochs', type=int, default=3,
                        help='Number of fine-tuning epochs after pruning. Defaults to 3.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility. Defaults to 42.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable detailed logging and visualization.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
