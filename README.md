# README

# Speaker Verification Based on VGGVox Model

A speaker verification framework based on Tensorflow 2.2 VggVox.

This project applied vggvox's speaker identification to vggvox's speaker verification.
Tensorflow 2.2 is also applied for faster R&D process by leverage `tf.data.Dataset`.
A new research, develop and deployment framework, which is called `DFM-TES`, is introduced. 
It is a research-to-production pipeline framework focused on AI/ML engineering.
The DFM-TES is an abbreviation of
- Data - It's define data model, data source adapter, and dataset adapter components.
- Feature - It's define feature extraction and transformation for pipelines.
- Model - It's define AI/ML models in pipeline.
- Training - It's define how to train the model or feature extraction given a training dataset.
- Evaluation - It's define how to evaluate and measurement the front-end, back-end, and end-2-end pipeline. 
Experiment's metrics and visualizations may define as empirical evaluation.
- Serving - It's define environments and deployment methods to serve models for training, evaluation, research, and production.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

**Prerequisites**
Install these software for building and launching.

- [Python 3.7.7](https://www.python.org/downloads/release/python-377/)
- [Anaconda 4.7.12](https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh)
- pip 20.0.2
- Ubuntu 18.04.4

**Installing**
Install all aboved prerequisites. Then install by following steps.

#### 1. Clone the project to your local repository.
```shell script
GIT_URL=https://github.com/rattaphon.h
git clone ${GIT_URL}/speaker-verification-vggvox
```

#### 2. Create virtual environment
```shell script
cd speaker-verification-vggvox
conda create -n vggvox37tf2 -python=3.7.7
```

#### 3. Install `ffmpeg` library
```shell script
bash ./install_decoder.sh
```

#### 4. Install dependencies
```shell script
bash ./install_decoder.sh
```


## Running the tests


```

```

## Deployment


```

```
- Use as a lib, please see API list. [under development]

## Built With

## Speaker Verification

## Contributing

All contribution are gracefully accept.

## Versioning

We use [SemVer 2.0.0](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ratthapon/fractal-compression/tags).

## Authors
- **Rattaphon Hokking** - *Initial work* - [ratthapon](https://github.com/ratthapon)

See also the list of [contributors](https://github.com/ratthapon/speaker-verification-vggvox/graphs/contributors) who participated in this project.

## Acknowledgments

## Issues

## References

## License


