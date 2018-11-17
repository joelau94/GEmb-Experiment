import argparse
import os

import experiment


def main():
  parser = argparse.ArgumentParser(description='GEmb')

  parser.add_argument('--config-file', type=str,
                      default='../models/config.pkl')
  parser.add_argument('--use-gemb', dest='use_gemb', action='store_true')

  args = parser.parse_args()

  cfg = experiment.Config()
  cfg.load(args.config_file)

  exp = experiment.Experiment(cfg.config)
  exp.test(args.use_gemb)


if __name__ == '__main__':
  main()
