# BOOM (Benchmark of Observability Metrics) Evaluations

This directory contains example code for evaluating zero-shot foundation models as well as classical baselines against BOOM. For more information on the dataset, see the [dataset card](https://huggingface.co/datasets/Datadog/BOOM) in Hugging Face.

To run evals for Toto, make sure you've followed the installation instructions in this repository.

## Models

- [Toto (this repository)](https://github.com/DataDog/toto)
- [Toto 2.0 (this repository)](https://github.com/DataDog/toto/tree/main/toto2)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Chronos-2](https://github.com/amazon-science/chronos-forecasting)
- [Moirai](https://github.com/SalesforceAIResearch/uni2ts)
- [Moirai 2.0](https://github.com/SalesforceAIResearch/uni2ts)
- [TimesFM](https://github.com/google-research/timesfm)
- [TimesFM 2.5](https://github.com/google-research/timesfm)
- [VisionTS](https://github.com/Keytoyze/VisionTS.git)
- [Timer](https://github.com/thuml/Large-Time-Series-Model.git)
- [Time-MoE](https://github.com/Time-MoE/Time-MoE.git)
- [Auto-ARIMA, Auto-ETS, Auto-Theta, Seasonal Naive](https://github.com/SalesforceAIResearch/gift-eval) (included in the Gift-Eval repository)


Our evaluation methodology is adapted from [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval). To run these notebooks for each model, you will need to install Gift-Eval as well as the required environment for each model.

### Toto
To set up the environment for Toto, follow the instructions in the [README](/README.md).

### Toto 2.0
To set up the environment for Toto 2.0, follow the Toto 2.0 installation instructions in the [README](/README.md). Register the resulting venv as the `toto2_eval_env` Jupyter kernel so the `toto2.ipynb` notebook binds to it.

Download the following environments to reproduce these notebooks:

```sh
mkdir /notebook_env
curl -L https://github.com/SalesforceAIResearch/uni2ts/archive/cadebd82106e32409b7854b033dbd7a68de87fc0.tar.gz -o /notebook_env/moirai.tar.gz

curl -L https://github.com/amazon-science/chronos-forecasting/archive/6166d284f467da7befc206f6a5b6b2bc1a794a87.tar.gz -o /notebook_env/chronos.tar.gz

curl -L https://github.com/google-research/timesfm/archive/9594c0618dec116e5006ef71a3d7f19630e00a0c.tar.gz -o /notebook_env/timesfm.tar.gz

curl -L https://github.com/Time-MoE/Time-MoE/archive/8ce3c93898ca13fe05449370c0ff372a79711a47.tar.gz -o /notebook_env/time-moe.tar.gz

curl -L https://github.com/Keytoyze/VisionTS/archive/9fc5f32311c161504e0a2be0f3c8f7f29e41923e.tar.gz -o /notebook_env/visionts.tar.gz

curl -L https://github.com/thuml/Large-Time-Series-Model/archive/fee65cb8fbd0a1474a23829d68e9e2ed23ff16ab.tar.gz -o /notebook_env/timer.tar.gz

curl -L https://github.com/SalesforceAIResearch/uni2ts/archive/8062ef5a5660d2fea395fd1288ec9c397396c168.tar.gz -o /notebook_env/moirai2.tar.gz

curl -L https://github.com/amazon-science/chronos-forecasting/archive/fd533389c300660f9d8e3a00fcb29e4ca1174745.tar.gz -o /notebook_env/chronos2.tar.gz

curl -L https://github.com/google-research/timesfm/archive/6ae67d41d813fcdab0a1bc785b79053c3769a63e.tar.gz -o /notebook_env/timesfm2_5.tar.gz
```

> **Moirai 2.0 note:** The `moirai2.ipynb` notebook's first code cell
> re-pins `gluonts~=0.14.3` because Gift-Eval's pyproject upgrades gluonts
> to 0.15.1, which breaks `QuantileForecastGenerator` for Moirai 2.0. No
> manual action required — just run the notebook top to bottom.

> **Python version:** use Python **3.11**. Gift-Eval's pinned SHA
> requires `pandas==2.0.0` (exact), which has no wheels for Python 3.12+
> on Apple Silicon / some Linux arches.

After downloading these repos, intialize a virtual environment for each model:
```sh
MODEL_NAME = #change this accordingly
mkdir -p "/venvs/${MODEL_NAME}_eval_env"
python -m venv "/venvs/${MODEL_NAME}_eval_env"
source "/venvs/${MODEL_NAME}_eval_env/bin/activate"
```

Then follow the installation instructions within each repository for environment setup.

After setting up the model specific environment, we then install Gift-Eval for dataloading and processing
```sh
curl -L https://github.com/SalesforceAIResearch/gift-eval/archive/1527c41589189ad1bc3883ed4d3d97b3e5a3b47c.tar.gz -o /notebook_env/gift-eval.tar.gz
```

Follow Gift-Eval instructions to setup environment on top of the model environment. Note: for the statistical baselines like Auto-ARIMA, Auto-ETS, etc. that depend on StatsForecast, all the necessary dependencies are included in Gift-Eval when you install with `pip install -e .[baseline]`

Finally, setup the environment for notebooks:
```sh
pip install --upgrade-strategy only-if-needed ipykernel
python -m ipykernel install --user --name "${MODEL_NAME}_eval_env" --display-name "${MODEL_NAME}_eval_env" || echo "Warning: Failed to install Jupyter kernel for $MODEL_NAME"
```
