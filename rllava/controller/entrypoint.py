"""Common construction and launch path for all online RL entry points."""

from rllava.controller.config import RLConfig


def main():
    from rllava.utils.config import init_config
    from rllava.utils.tokenizer import load_tokenizer_and_processor
    from rllava.data.data_loader import create_dataloader
    from .controller import Controller

    config = init_config(RLConfig)
    tokenizer, processor = load_tokenizer_and_processor(
        config.actor.model.model_path, config.data.override_chat_template
    )
    train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)
    controller = Controller(config, tokenizer, processor, train_dataloader, val_dataloader)
    try:
        controller.run()
    finally:
        controller.close()


if __name__ == "__main__":
    from rllava.distribute.dist_utils import init_dist
    init_dist()
    main()
