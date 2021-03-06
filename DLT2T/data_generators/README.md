# T2T Problems.

This directory contains `Problem` specifications for a number of problems. We
use a naming scheme for the problems, they have names of the form
`[task-family]_[task]_[specifics]`.  Data for all currently supported problems
can be generated by calling the main generator binary (`t2t-datagen`). For
example:

```
t2t-datagen \
  --problem=algorithmic_identity_binary40 \
  --data_dir=/tmp
```

will generate training and development data for the algorithmic copy task -
`/tmp/algorithmic_identity_binary40-dev-00000-of-00001` and
`/tmp/algorithmic_identity_binary40-train-00000-of-00001`.
All tasks produce TFRecord files of `tensorflow.Example` protocol buffers.


## Adding a new problem

To add a new problem, subclass
[`Problem`](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/problem.py)
and register it with `@registry.register_problem`. See
[`WMTEnDeTokens8k`](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/wmt.py)
for an example.

`Problem`s support data generation, training, and decoding.

Data generation is handled by `Problem.generate_data` which should produce 2
datasets, training and dev, which should be named according to
`Problem.training_filepaths` and `Problem.dev_filepaths`.
`Problem.generate_data` should also produce any other files that may be required
for training/decoding, e.g. a vocabulary file.

A particularly easy way to implement `Problem.generate_data` for your dataset is
to create 2 Python generators, one for the training data and another for the
dev data, and pass them to `generator_utils.generate_dataset_and_shuffle`. See
[`WMTEnDeTokens8k.generate_data`](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/wmt.py)
for an example of usage.

The generators should yield dictionaries with string keys and values being lists
of {int, float, str}.  Here is a very simple generator for a data-set where
inputs are lists of 2s with length upto 100 and targets are lists of length 1
with an integer denoting the length of the input list.

```
def length_generator(nbr_cases):
  for _ in xrange(nbr_cases):
    length = np.random.randint(100) + 1
    yield {"inputs": [2] * length, "targets": [length]}
```

Note that our data reader uses 0 for padding and other parts of the code assume
end-of-string (EOS) is 1, so it is a good idea to never generate 0s or 1s,
except if all your examples have the same size (in which case they'll never be
padded anyway) or if you're doing padding on your own (in which case please use
0s for padding). When adding the python generator function, please also add unit
tests to check if the code runs.

The generator can do arbitrary setup before beginning to yield examples - for
example, downloading data, generating vocabulary files, etc.

Some examples:

*   [Algorithmic problems](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/algorithmic.py)
    and their [unit tests](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/algorithmic_test.py)
*   [WMT problems](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/wmt.py)
    and their [unit tests](https://github.com/tensorflow/DLT2T/tree/master/DLT2T/data_generators/wmt_test.py)
