from cores.configs import parse_config
from cores.pretrain_trainer import Pretrainer

def main():
    config = parse_config()

    # 可选：打印配置
    print("Final Configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    # 创建 trainer 并训练
    trainer = Pretrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
