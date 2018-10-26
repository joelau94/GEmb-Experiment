import argparse
import os

import experiment


def main():
  parser = argparse.ArgumentParser(description='GEmb')

  parser.add_argument('--config-file', type=str,
                      default='../models/config.pkl')

  args = parser.parse_args()

  cfg = experiment.Config()
  cfg.load(args.config_file)

  exp = experiment.Experiment(cfg.config)
  exp.test()


if __name__ == '__main__':
  main()
