import sys

def convert(in_file, out_file):
  fin = open(in_file, 'r')
  fout = open(out_file, 'w')

  words, tags = [], []
  for line in fin:
    if line.strip() == '':
      if len(words) > 0 and len(words) == len(tags):
        fout.write('{} ||| {}\n'.format(' '.join(words), ' '.join(tags)))
      words, tags = [], []
    else:
      line = line.strip().split('\t')
      words.append(line[1].lower())
      tags.append(line[3])


def main():
  convert(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
  main()
