# DC-LA

To obtain histograms of the samples from the three samplers in the 2D experiment, run the following.
**Use multiple Markov chains and retain only the final samples from each chain.**
```bash
python l12_exp2.py
```
**Use one chain, discarding burn-in samples**
```bash
python l12_exp.py
```
## Compressed Sensing experiments

Gaussian sensing matrix
```bash
python CS_exp.py --sensing gaussian
```

Partial DCT sensing matrix
```bash
python CS_exp.py --sensing pdct
```

Oversampled DCT sensing matrix with oversample factor being, e.g., 5
```bash
python CS_exp.py --sensing odct --oversample_factor 5
```
