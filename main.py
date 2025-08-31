from configs.pretrain_config import parse_pretrain_config
from cores.pretrain_trainer import Pretrainer


def main():
    config = parse_pretrain_config()

    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    trainer = Pretrainer(config)
    trainer.train()
    trainer.register_from_loaders()


if __name__ == '__main__':
    main()
