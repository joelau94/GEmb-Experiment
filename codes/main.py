import argparse
import cPickle as pkl
import os

import experiment


def main():
  parser = argparse.ArgumentParser(description='GEmb')

  parser.add_argument('--config-file', type=str,
                      default='../models/config.pkl')
  parser.add_argument('--start-over', type=bool, default=False)
  parser.add_argument('--train-gemb', type=bool, default=False)

  parser.add_argument('--train-data-file', type=str,
                      default='../data/train.pkl')
  parser.add_argument('--dev-data-file', type=str,
                      default='../data/dev.pkl')
  parser.add_argument('--test-data-file', type=str,
                      default='../data/test.pkl')

  parser.add_argument('--task', type=str, default='tagging',
                      choices=['tagging', 'classification'])
  parser.add_argument('--use-gemb', type=bool, default=True)
  parser.add_argument('--keep-prob', type=float, default=0.9)

  parser.add_argument('--dictfile', type=str,
                      default='../data/dicts.pkl')
  parser.add_argument('--embed-dim', type=int, default=300)
  parser.add_argument('--hidden-dims', type=int, nargs='+',
                      default=[256,256])

  parser.add_argument('--seed', type=int, default=23)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--beta1', type=float, default=0.9)
  parser.add_argument('--beta2', type=float, default=0.99)
  parser.add_argument('--eps', type=float, default=1e-8)
  parser.add_argument('--clip-norm', type=float, default=5.0)

  parser.add_argument('--ckpt', type=str,
                      default='../models/model')
  parser.add_argument('--max-ckpts', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--max-steps', type=int, default=1000000)
  parser.add_argument('--gemb-steps', type=int, default=1000000)
  parser.add_argument('--print-interval', type=int, default=50)
  parser.add_argument('--save-interval', type=int, default=1000)

  args = parser.parse_args()

  dicts = pkl.load(open(args.dictfile, 'rb'))
  vocab_size = len(dicts['i2w'])
  num_class = len(dicts['i2t'])

  cfg = experiment.Config()
  if not args.start_over and os.path.exists(args.config_file):
    cfg.load(args.config_file)
  else:
    cfg.config = {
        'train_data_file': args.train_data_file,
        'dev_data_file': args.dev_data_file,
        'test_data_file': args.test_data_file,

        'task': args.task,
        'use_gemb': args.use_gemb,
        'keep_prob': args.keep_prob,

        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'hidden_dims': args.hidden_dims,
        'num_class': num_class,

        'seed': args.seed,
        'lr': args.lr,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'clip_norm': args.clip_norm,

        'ckpt': args.ckpt,
        'max_ckpts': args.max_ckpts,
        'batch_size': args.batch_size,
        'max_steps': args.max_steps,
        'gemb_steps': args.gemb_steps,
        'print_interval': args.print_interval,
        'save_interval': args.save_interval
    }

    if not os.path.isdir(os.path.dirname(args.config_file)):
      os.mkdir(os.path.dirname(args.config_file))
    cfg.save(args.config_file)

  exp = experiment.Experiment(cfg.config)
  if args.train_gemb:
    exp.train_gemb()
  else:
    exp.train()


if __name__ == '__main__':
  main()
