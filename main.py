from configs.pretrain_config import parse_pretrain_config
from configs.adapt_config import parse_adaption_config
from cores.pretrain_trainer import Pretrainer
from downstream.adapt_trainer import AdaptTrainer


def main():
    # config = parse_pretrain_config()
    #
    # print("Final Configuration:")
    # for k, v in config.__dict__.items():
    #     print(f"  {k}: {v}")
    #
    # trainer = Pretrainer(config)
    # trainer.train()
    # trainer.register_from_loaders()

    config = parse_adaption_config()

    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    trainer = AdaptTrainer(config)
    trainer.train()



if __name__ == '__main__':
    main()
