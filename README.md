# Quantum-inspired binary classification

This is an implementation of the Helstrom Quantum Centroid Simulation(HQCS) classifier introduced in this [link][5].
The works in this repository provides python codes that allow users to regenerate experimental data provided in the paper.

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

`max_copies` sets the upper bound of quantum copies, `name` is used when saving the classification score and f1 score in the `results` folder. `start` sets the starting point of the number of quantum copies, and `step_size` sets the increment of quantum copies. In this case, we are running the HQCS from quantum copies one to hundred with increment of 0.25.

The output `f1_scores` is a `dictionary` of form:
```python
{'f1_hel': 1.0, 'copies_hel': 0.25, 'f1_fid': 1.0, 'copies_fid': 0.75}
```
`f1_hel` and `f1_fid` are the maximum f1 score achieved in the given boundary of quantum copies.
`copies_hel` and `copies_fid` are corresponding number of quantum copies at which the maximum occurs

## Classical classifiers

In the `classical_classifiers` folder, we provide python code that uses  [ax-platform][3] to optimize hyperparameters of classical supervised learning models. The code is capable of optimizing the hyperparameters of 13 standard supervised learning models including XGBoost, CatBoost, and Random Forest Classifiers etc.

For example, to perform hyperparameter optimization on `Parkinson` dataset for XGBoost and Random Forest Classifiers, we first load the data set:

```python
from helstrom_classifier.load_data import load_datasets
X, y  = load_datasets('parkinson')
```
Then, we define appropriate hyperparameter search space:
```python

parameters = {
 "XGBooster": [
    {"name": "learning_rate", "type": "range", "bounds": [0.01, 0.2]},
    {"name": "verbosity", "type": "choice", "values": [0, 1, 2, 3]},
    {"name": "booster", "type": "choice", "values": ["gbtree", "dart"]},
    {"name": "gamma", "type": "range", "bounds": [0.1, 1.0]},
    {"name": "reg_alpha", "type": "range", "bounds": [40, 180]},
    {"name": "min_child_weight", "type": "range", "bounds": [0, 10]}
    ],
"RandomForest": [
    {"name": "n_estimators", "type": "range", "bounds": [50, 200]},
    {"name": "max_depth", "type": "range", "bounds": [5, 20]},
    {"name": "min_samples_split", "type": "range", "bounds": [2, 10]},
    {"name": "boostrap", "type": "choice", "values": [True, False]}
    ]
}

```
Finally, we perform the hyperparameter optimization:
```python
from classical_classifiers.optimize_loop import optimize_parameters

df = optimize_parameters(parameters, classifiers, X, y)

```
`df` is in `DataFrame` form:

| Classifiers| f1_score |
| -------- | -------- | 
| RandomForest   | 0.8829  | 
| XGBoost    | 0.9001   | 

and it will be saved in `output/classical_output` folder.

## Plots

In the `plots` folder, we provide python codes that were used to generate plots in the paper.

## Dataset Classification

In the `Dataset Classification` folder, we provide python code that characterizes the strucutre of a given dataset based on the methodology explained in [Types of minority class examples and their influence on learning classifiers from imbalanced data][4]. This corresponds to the `Appendix A: Classification of datasets` in our paper.



[1]: https://arxiv.org/abs/2012.00058L
[2]: https://archive.ics.uci.edu/about
[3]: https://ax.dev/
[4]: https://link.springer.com/article/10.1007/s10844-015-0368-1
[5]: https://arxiv.org/abs/2403.15308
