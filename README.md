<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

A small Python package for preparing *ordered* language data for RNN language models.

The user can chose white-space tokenization or Byte-Level BPE tokenization, provided by the `tokenizers` package.

## Usage

```python
from preppy import Prep

sentences = ['Hello World.', 'Hello World.']

prep = Prep(sentences,
            reverse=False,  # generate batches starting from last document
            num_types=2,    # if not None, create B-BPE tokenizer
            batch_size=1,   # batch size 
            context_size=1, # number of back-prop-through-time steps
            sliding=False,  # windows slide over input text
            )
            
for batch in prep.generate_batches():
   pass  # train model on batch
```

## Compatibility

Developed on Ubuntu 18.04 and Python 3.7
