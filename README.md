# A benchmark for time-frequency denoising/ detecting methods
[![forthebadge](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNjUuNzkiIGhlaWdodD0iMzUiIHZpZXdCb3g9IjAgMCAxNjUuNzkgMzUiPjxyZWN0IGNsYXNzPSJzdmdfX3JlY3QiIHg9IjAiIHk9IjAiIHdpZHRoPSI2NS4xMyIgaGVpZ2h0PSIzNSIgZmlsbD0iIzMxQzRGMyIvPjxyZWN0IGNsYXNzPSJzdmdfX3JlY3QiIHg9IjYzLjEzIiB5PSIwIiB3aWR0aD0iMTAyLjY2IiBoZWlnaHQ9IjM1IiBmaWxsPSIjMzg5QUQ1Ii8+PHBhdGggY2xhc3M9InN2Z19fdGV4dCIgZD0iTTE2LjUxIDIyTDEzLjQ2IDEzLjQ3TDE1LjA4IDEzLjQ3TDE3LjIyIDIwLjE0TDE5LjM5IDEzLjQ3TDIxLjAyIDEzLjQ3TDE3Ljk1IDIyTDE2LjUxIDIyWk0yNi41MyAyMkwyNS4wNiAyMkwyNS4wNiAxMy40N0wyNi41MyAxMy40N0wyNi41MyAyMlpNMzYuOTIgMjJMMzEuMzQgMjJMMzEuMzQgMTMuNDdMMzYuODggMTMuNDdMMzYuODggMTQuNjZMMzIuODMgMTQuNjZMMzIuODMgMTcuMDJMMzYuMzMgMTcuMDJMMzYuMzMgMTguMTlMMzIuODMgMTguMTlMMzIuODMgMjAuODJMMzYuOTIgMjAuODJMMzYuOTIgMjJaTTQyLjUwIDIyTDQwLjUzIDEzLjQ3TDQyLjAwIDEzLjQ3TDQzLjMzIDE5Ljg4TDQ0Ljk2IDEzLjQ3TDQ2LjIwIDEzLjQ3TDQ3LjgxIDE5Ljg5TDQ5LjEyIDEzLjQ3TDUwLjU5IDEzLjQ3TDQ4LjYyIDIyTDQ3LjIxIDIyTDQ1LjU4IDE1Ljc3TDQzLjkyIDIyTDQyLjUwIDIyWiIgZmlsbD0iI0ZGRkZGRiIvPjxwYXRoIGNsYXNzPSJzdmdfX3RleHQiIGQ9Ik03OS43MCAyMkw3Ny4zMiAyMkw3Ny4zMiAxMy42MEw4MS4xNiAxMy42MFE4Mi4zMCAxMy42MCA4My4xNCAxMy45OFE4My45OCAxNC4zNSA4NC40NCAxNS4wNlE4NC44OSAxNS43NiA4NC44OSAxNi43MUw4NC44OSAxNi43MVE4NC44OSAxNy42MiA4NC40NyAxOC4zMFE4NC4wNCAxOC45OCA4My4yNSAxOS4zNkw4My4yNSAxOS4zNkw4NS4wNiAyMkw4Mi41MiAyMkw4MC45OSAxOS43N0w3OS43MCAxOS43N0w3OS43MCAyMlpNNzkuNzAgMTUuNDdMNzkuNzAgMTcuOTNMODEuMDIgMTcuOTNRODEuNzUgMTcuOTMgODIuMTIgMTcuNjFRODIuNDkgMTcuMjkgODIuNDkgMTYuNzFMODIuNDkgMTYuNzFRODIuNDkgMTYuMTIgODIuMTIgMTUuNzlRODEuNzUgMTUuNDcgODEuMDIgMTUuNDdMODEuMDIgMTUuNDdMNzkuNzAgMTUuNDdaTTk2LjQzIDIyTDg5LjY4IDIyTDg5LjY4IDEzLjYwTDk2LjI3IDEzLjYwTDk2LjI3IDE1LjQ0TDkyLjA0IDE1LjQ0TDkyLjA0IDE2Ljg1TDk1Ljc3IDE2Ljg1TDk1Ljc3IDE4LjYzTDkyLjA0IDE4LjYzTDkyLjA0IDIwLjE3TDk2LjQzIDIwLjE3TDk2LjQzIDIyWk0xMDAuNjQgMjEuMjRMMTAwLjY0IDIxLjI0TDEwMS40MiAxOS40OVExMDEuOTkgMTkuODYgMTAyLjczIDIwLjA5UTEwMy40NyAyMC4zMiAxMDQuMTkgMjAuMzJMMTA0LjE5IDIwLjMyUTEwNS41NiAyMC4zMiAxMDUuNTcgMTkuNjRMMTA1LjU3IDE5LjY0UTEwNS41NyAxOS4yOCAxMDUuMTggMTkuMTFRMTA0Ljc5IDE4LjkzIDEwMy45MiAxOC43NEwxMDMuOTIgMTguNzRRMTAyLjk3IDE4LjUzIDEwMi4zMyAxOC4zMFExMDEuNzAgMTguMDYgMTAxLjI0IDE3LjU1UTEwMC43OSAxNy4wMyAxMDAuNzkgMTYuMTZMMTAwLjc5IDE2LjE2UTEwMC43OSAxNS4zOSAxMDEuMjEgMTQuNzdRMTAxLjYzIDE0LjE1IDEwMi40NiAxMy43OVExMDMuMzAgMTMuNDMgMTA0LjUxIDEzLjQzTDEwNC41MSAxMy40M1ExMDUuMzMgMTMuNDMgMTA2LjE0IDEzLjYyUTEwNi45NCAxMy44MCAxMDcuNTYgMTQuMTdMMTA3LjU2IDE0LjE3TDEwNi44MyAxNS45M1ExMDUuNjIgMTUuMjggMTA0LjQ5IDE1LjI4TDEwNC40OSAxNS4yOFExMDMuNzggMTUuMjggMTAzLjQ2IDE1LjQ5UTEwMy4xNCAxNS43MCAxMDMuMTQgMTYuMDRMMTAzLjE0IDE2LjA0UTEwMy4xNCAxNi4zNyAxMDMuNTIgMTYuNTRRMTAzLjkxIDE2LjcxIDEwNC43NiAxNi44OUwxMDQuNzYgMTYuODlRMTA1LjcyIDE3LjEwIDEwNi4zNSAxNy4zM1ExMDYuOTggMTcuNTYgMTA3LjQ0IDE4LjA3UTEwNy45MCAxOC41OCAxMDcuOTAgMTkuNDZMMTA3LjkwIDE5LjQ2UTEwNy45MCAyMC4yMSAxMDcuNDggMjAuODNRMTA3LjA3IDIxLjQ0IDEwNi4yMyAyMS44MFExMDUuMzggMjIuMTcgMTA0LjE4IDIyLjE3TDEwNC4xOCAyMi4xN1ExMDMuMTYgMjIuMTcgMTAyLjIwIDIxLjkyUTEwMS4yNCAyMS42NyAxMDAuNjQgMjEuMjRaTTExMi40MSAxOC4yNkwxMTIuNDEgMTguMjZMMTEyLjQxIDEzLjYwTDExNC43OSAxMy42MEwxMTQuNzkgMTguMTlRMTE0Ljc5IDIwLjIwIDExNi4zOCAyMC4yMEwxMTYuMzggMjAuMjBRMTE3Ljk2IDIwLjIwIDExNy45NiAxOC4xOUwxMTcuOTYgMTguMTlMMTE3Ljk2IDEzLjYwTDEyMC4zMSAxMy42MEwxMjAuMzEgMTguMjZRMTIwLjMxIDIwLjEzIDExOS4yNyAyMS4xNVExMTguMjMgMjIuMTcgMTE2LjM2IDIyLjE3TDExNi4zNiAyMi4xN1ExMTQuNDggMjIuMTcgMTEzLjQ1IDIxLjE1UTExMi40MSAyMC4xMyAxMTIuNDEgMTguMjZaTTEzMS43OCAyMkwxMjUuMzkgMjJMMTI1LjM5IDEzLjYwTDEyNy43NyAxMy42MEwxMjcuNzcgMjAuMTFMMTMxLjc4IDIwLjExTDEzMS43OCAyMlpNMTM3Ljk5IDE1LjQ4TDEzNS40MSAxNS40OEwxMzUuNDEgMTMuNjBMMTQyLjkzIDEzLjYwTDE0Mi45MyAxNS40OEwxNDAuMzYgMTUuNDhMMTQwLjM2IDIyTDEzNy45OSAyMkwxMzcuOTkgMTUuNDhaTTE0Ni43MiAyMS4yNEwxNDYuNzIgMjEuMjRMMTQ3LjUwIDE5LjQ5UTE0OC4wNiAxOS44NiAxNDguODAgMjAuMDlRMTQ5LjU1IDIwLjMyIDE1MC4yNyAyMC4zMkwxNTAuMjcgMjAuMzJRMTUxLjYzIDIwLjMyIDE1MS42NCAxOS42NEwxNTEuNjQgMTkuNjRRMTUxLjY0IDE5LjI4IDE1MS4yNSAxOS4xMVExNTAuODYgMTguOTMgMTQ5Ljk5IDE4Ljc0TDE0OS45OSAxOC43NFExNDkuMDQgMTguNTMgMTQ4LjQxIDE4LjMwUTE0Ny43NyAxOC4wNiAxNDcuMzIgMTcuNTVRMTQ2Ljg2IDE3LjAzIDE0Ni44NiAxNi4xNkwxNDYuODYgMTYuMTZRMTQ2Ljg2IDE1LjM5IDE0Ny4yOCAxNC43N1ExNDcuNzAgMTQuMTUgMTQ4LjU0IDEzLjc5UTE0OS4zNyAxMy40MyAxNTAuNTggMTMuNDNMMTUwLjU4IDEzLjQzUTE1MS40MCAxMy40MyAxNTIuMjEgMTMuNjJRMTUzLjAyIDEzLjgwIDE1My42MyAxNC4xN0wxNTMuNjMgMTQuMTdMMTUyLjkwIDE1LjkzUTE1MS43MCAxNS4yOCAxNTAuNTcgMTUuMjhMMTUwLjU3IDE1LjI4UTE0OS44NiAxNS4yOCAxNDkuNTMgMTUuNDlRMTQ5LjIxIDE1LjcwIDE0OS4yMSAxNi4wNEwxNDkuMjEgMTYuMDRRMTQ5LjIxIDE2LjM3IDE0OS42MCAxNi41NFExNDkuOTggMTYuNzEgMTUwLjgzIDE2Ljg5TDE1MC44MyAxNi44OVExNTEuNzkgMTcuMTAgMTUyLjQyIDE3LjMzUTE1My4wNSAxNy41NiAxNTMuNTEgMTguMDdRMTUzLjk4IDE4LjU4IDE1My45OCAxOS40NkwxNTMuOTggMTkuNDZRMTUzLjk4IDIwLjIxIDE1My41NiAyMC44M1ExNTMuMTQgMjEuNDQgMTUyLjMwIDIxLjgwUTE1MS40NiAyMi4xNyAxNTAuMjYgMjIuMTdMMTUwLjI2IDIyLjE3UTE0OS4yNCAyMi4xNyAxNDguMjcgMjEuOTJRMTQ3LjMxIDIxLjY3IDE0Ni43MiAyMS4yNFoiIGZpbGw9IiNGRkZGRkYiIHg9Ijc2LjEzIi8+PC9zdmc+)](./results/readme.md)

## Summary

- [A benchmark for time-frequency denoising/ detecting methods](#a-benchmark-for-time-frequency-denoising-detecting-methods)
  - [Summary](#summary)
  - [What is this benchmark?](#what-is-this-benchmark)
  - [How to use this benchmark?](#how-to-use-this-benchmark)
    - [Cloning this repository](#cloning-this-repository)
    - [Using a template file for your method](#using-a-template-file-for-your-method)
    - [Adding your method's dependencies with ```poetry```](#adding-your-methods-dependencies-with-poetry)
    - [Checking everything is in order with ```pytest```](#checking-everything-is-in-order-with-pytest)
    - [Branching the repository](#branching-the-repository)
    - [Running this benchmark locally](#running-this-benchmark-locally)
  - [Standard benchmark for denoising methods](#standard-benchmark-for-denoising-methods)
  - [Standard benchmark for detection methods](#standard-benchmark-for-detection-methods)

## What is this benchmark?

A benchmark is a comparison between different methods when running an standardized test. The goal of this benchmark is to compare different methods for denoising / detecting a signal based on different characterizations of the time-frequency representation of the signal. In particular, our goal is to evaluate the performance of techniques based on the zeros of the spectrogram and to contrast them with more traditional methods, like those based on the ridges of that time-frequency distribution.

Nevertheless, the methods to compare, the tests, and the performance evaluation functions were conceived as different modules, so that one can assess new methods without modifying the tests or the signals. On the one hand, the tests and the performance evaluation functions are encapsulated in the class `Benchmark`. On the other hand, the signals used in this benchmark are generated by the methods in the class `SignalBank`. The only restriction this poses is that the methods should satisfy some requirements regarding the *shape of their input an output parameters*.

## How to use this benchmark?

You can use this benchmark to test a new method against others. There are at least two ways of doing this. You can either make a new branch of this repository and push a new method to test, or you can clone this repository and benchmark your own method locally, i.e. in your computer. In the first case, a workflow using GitHub actions will automatically detect your new method and run the standard benchmark tests. In the second case you can run the benchmark with all the modifications you need.

The [*notebooks*](./notebooks/) folder contains a number of minimal working examples to understand how this benchmark works and how you could use it for your project. In particular, [*demo_benchmark.ipynb*](./notebooks/demo_benchmark.ipynb) gives two minimal working examples to introduce the basic functionality of the `Benchmark` class, and the notebook [*demo_signal_bank.ipynb*](./notebooks/demo_signal_bank.ipynb) showcases the signals produced by the `SignalBank` class.

The instructions below will help you to add a new method and run the benchmark afterwards.

### Cloning this repository

First you should have a local copy of this repository to add and modify files. For this, open a terminal in the directory you prefer and use:

```bash
git clone https://github.com/jmiramont/benchmark-test.git
```

A new method can be tested against others simply by adding a file that contains it into the folder [src/methods](./src/methods). To do this, you method must first have the following signature:

```python
    def a_new_method(signals, params):
        ...
```

Methods should receive an `M`x`N` numpy array of signals, where `M` is the number of signals, and `N` is the number of their time samples. Additionally, they should receive a second parameter `params` to allow testing different combinations of input parameters. The shape of the output depends on the task (*denoising* or *detection*). The output of your method must be of a certain shape and type, regarding the task your method is devoted to:

- For Denoising: The output must be a numpy array and have the same shape as the input (an array of shape `M`x`N`).
- For Detection: The output must be an array whose first dimension is equal to `M`.

Now you have to add a file with your method in the folder [src/methods](./src/methods/). Let's see how to do this in the following section.

### Using a template file for your method

The name of the file with your method must start with *method_* and have certain content to help the benchmark to automatically discover new added methods:

1. The file should encapsulate your method in a new class that inherits from a template called `NewMethod`.
2. The file must include a the definition of a function `instantiate_method()` that instantiates an object of the class that represent your method.

This is much easier than it sounds :). To make it simpler, [a file called *method_new_basic_template.py* is provided](./new_method_example/method_new_basic_template.py) which you can use as a template. You just have to fill in the parts that implement your method.

Let's take a look at the different parts of the template file  *method_new_basic_template.py*. In the first section of the template file you can either import a function with your method, or implement everything in this file:

```python
from methods.MethodTemplate import MethodTemplate
# You must import the MethodTemplate abstract class.

"""
|
| Import here all the modules you need.
| Remark: Make sure that neither of those modules starts with "method_".
|
"""

""" 
|
| Put here all the functions that your method uses.
| Remark: Make sure that this file starts with "method_".
|
| def a_function_of_my_method(signal,params):
|   ...
|
"""
```

In the next part of the file, your method is encapsulated in a new class called `NewMethod` (you can change this name if you prefer to). The only requisite for the class that represents your method is that it inherits from the [abstract class](https://docs.python.org/3/library/abc.html) `MethodTemplate`. Abstract classes are not implemented, but they serve the purpose of establishing a template for new classes by forcing the implementation of certain *abstract* methods. This simply means that you will have to implement the class constructor and a class method called -unsurprisingly- `method()`:

```python

""" Create here a new class that will encapsulate your method.
This class should inherit the abstract class MethodTemplate.
By doing this, you must then implement the class method: 

def method(self, signal, params)

which should receive the signals and any parameters
that you desire to pass to your method.You can use this file as an example.
"""

class NewMethod(MethodTemplate):
    def __init__(self):
      self.id = 'a_new_method'
      self.task = 'denoising'  # Should be either 'denoising' or 'detection'

    def method(self, signals, params = None): # Implement this method.
        ...

    # def get_parameters(self):            # Use it to parametrize your method.
    #     return [None,]      

```

The constructor function ```__init__(self)``` must initialize the attributes ```self.id``` and ```self.task```. The first is an string to identify your method in the benchmark. The second is the name of the task your method is devoted to. This can be either ```'denoising'``` or ```'detection'```. Notice that if you fail to use such names this will prevent you from benchmarking your method.

Lastly, as anticipated above, you have to implement the class function ```method(self, signals, params)```. This function may act as a wrapper of your method, i.e. you implement your method elsewhere and call it from this function, or you could implement it directly here, this is up to you :).

If you want to test your method using different sets of parameters, you can also implement the function `get_parameters()` to return a list with the desired input parameters (you can find an example of this [here](./new_method_example/method_new_with_parameters.py)).


*Remark: Do not modify the abstract class `MethodTemplate`*.

Finally, **you have to move the file** with all the modifications to the folder [/src/methods](./src/methods). Changing the name of the file is possible, but keep in mind that **the file's name must start with "*method_*" to be recognizable**.

### Adding your method's dependencies with ```poetry```

Your method might need particular modules as dependencies that are not currently listed in the dependencies of the default benchmark. You can add them using [```poetry```](https://python-poetry.org/docs/), a tool for dependency management and packaging in python. First install ```poetry``` following the steps described [here](https://python-poetry.org/docs/#installation). Once you're done with this, open a terminal in the directory where you clone the benchmark (or use the console in your preferred IDE) and make ```poetry``` create a virtual environment and install all the current dependencies of the benchmark:

```bash
poetry install 
```

After this, add all your dependencies by modifying the ```.toml``` file in the folder, under the ```[tool.poetry.dependencies]``` section. For example:

```bash
[tool.poetry.dependencies]
python = ">=3.8,<3.11"
numpy = "^1.22.0"
matplotlib = "^3.5.1"
pandas = "^1.3.5"
```

A more convenient and interactive way to do this interactively is by using ```poetry```, for example:

```bash
poetry add numpy
```

and following the instructions prompted in the console.

*Remark: Notice that the use of ```poetry``` for adding the dependencies of your packet is key for running the benchmark using [GitHub Actions](./.github/workflows), please consider this while adding your method.*

### Checking everything is in order with ```pytest```

Once your dependencies are ready, you should check that everything is in order using the ```pytest``` testing suit. To do this, simply run the following in a console located in your local version of the repository:

```bash
poetry run pytest
```

This will check a series of important points for running the benchmark online, mainly:

1. Your method class inherits the ```MethodTemplate``` abstract class.
2. The inputs and outputs of your method follows the required format according to the designated task.

Once the tests are passed, you can now either branch the repository or run the benchmark locally.

### Branching the repository

First, create a new branch using:

```bash
git branch new_branch
```

After this, you can upload your new branch with:

```bash
git push origin new_branch
```

This should create a new branch called ```new_branch```, that stems from the default repository. Once this is done, the benchmark is run remotely using GitHub Actions.

*Remark: Notice that ```pytest``` is also run again in this workflow. Therefore, keep in mind that if your method didn't pass the tests locally, it won't pass them at this stage either.*

### Running this benchmark locally

If you prefer to run the benchmark locally, you can use:

```bash
poetry run run_this_benchmark.py
```

## Standard benchmark for denoising methods

## Standard benchmark for detection methods
