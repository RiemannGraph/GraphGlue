from configs import parse_pretrain_config, parse_adaption_config
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

    config = parse_adaption_config()

    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    trainer = AdaptTrainer(config)
    trainer.train()



if __name__ == '__main__':
    main()
