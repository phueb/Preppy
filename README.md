<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

A small Python package for preparing *ordered* language data for RNN language models.

Tokenization is not included.

## Usage

```python
from preppy import Prep

sentences = ['Hello World.', 'Hello World.']

prep = Prep(sentences,
            reverse=False,  # generate batches starting from last document
            batch_size=1,   # batch size 
            context_size=1, # number of back-prop-through-time steps
            sliding=False,  # windows slide over input text
            )
            
for batch in prep.generate_batches():
   pass  # train model on batch
```

## Compatibility

Developed on Ubuntu 18.04 and Python 3.7
