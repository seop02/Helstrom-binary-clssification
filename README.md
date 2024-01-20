# Helstrom-simulation

This is an implementation of the Helstrom Quantum Centroid Simulation(HQCS) classifier introduced in link.
The works in this repository provides python codes that allows user to regenerate experimental data provided in the papaer.

## Helstrom Quantum Centroid Simulation (HQCS)

The python codes that simulates the statistics of the Helstrom measurement is in "helstrom_classifier/helstrom.py". 

"helstrom_classifier/load_data.py" loads the real world datasets for binary classification.

The available datasets are Aids, Appendicitis, Banana, Breast Cancer, Echocardiogram, Glass, Haberman, Heart disease, Hepatitis, Ionosphere, Iris, Lupus, Parkinson, Penguin, Transfusion, and Wine. Theses are the datasets from [Penn Machine Learning Benchmarks(PMLB)][1], and the [University of California, Irvine Machine Learning repository][2].

To run HQCS classifier on the Iris dataset, for example, you first load the dataset using load_datasets:

```python
from helstrom_classifier.load_data import load_datasets

X, y = load_datasets('iris')
```

Then, you run the HQCS by:

```python
from helstrom_classifier.helstrom import helstrom_simulator

f1_scores = helstrom_simulator(X=X, y=y, max_copies=100, d_type=torch.float64, name='iris', start=1, step_size=0.25)
```

`max_copies` sets the upper bound of quantum copies, `name` is used when saving the classification score and f1 score in `results` folder. `start` sets the starting point of the number of quantum copies, and `step_size` sets the increment of quantum copies. In this case, we are running the HQCS from quantum copies one to hundred with increment of 0.25.

[1]: https://arxiv.org/abs/2012.00058
[2]: https://archive.ics.uci.edu/about
