# BOOM (Benchmark of Observability Metrics) Evaluations

This directory contains example code for evaluating zero-shot foundation models as well as classical baselines against BOOM. For more information on the dataset, see the [dataset card](https://huggingface.co/datasets/Datadog/BOOM) in Hugging Face.

To run evals for Toto, make sure you've followed the installation instructions in this repository.

## Models

- [Toto (this repository)](https://github.com/DataDog/toto)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Moirai](https://github.com/SalesforceAIResearch/uni2ts)
- [TimesFM](https://github.com/google-research/timesfm)
- [VisionTS](https://github.com/Keytoyze/VisionTS.git)
- [Timer](https://github.com/thuml/Large-Time-Series-Model.git)
- [Time-MoE](https://github.com/Time-MoE/Time-MoE.git)
- [Auto-ARIMA, Auto-ETS, Auto-Theta, Seasonal Naive](https://github.com/SalesforceAIResearch/gift-eval) (included in the Gift-Eval repository)


Our evaluation methodology is adapted from [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval). To run these notebooks for each model, you will need to install Gift-Eval as well as the required environment for each model.

### Toto
To set up the environment for Toto, follow the instructions in the [README](/README.md).

Download the following environments to reproduce these notebooks:

```sh
mkdir /notebook_env
curl -L https://github.com/SalesforceAIResearch/uni2ts/archive/cadebd82106e32409b7854b033dbd7a68de87fc0.tar.gz -o /notebook_env/moirai.tar.gz

curl -L https://github.com/amazon-science/chronos-forecasting/archive/6166d284f467da7befc206f6a5b6b2bc1a794a87.tar.gz -o /notebook_env/chronos.tar.gz

curl -L https://github.com/google-research/timesfm/archive/9594c0618dec116e5006ef71a3d7f19630e00a0c.tar.gz -o /notebook_env/timesfm.tar.gz

curl -L https://github.com/Time-MoE/Time-MoE/archive/8ce3c93898ca13fe05449370c0ff372a79711a47.tar.gz -o /notebook_env/time-moe.tar.gz

curl -L https://github.com/Keytoyze/VisionTS/archive/9fc5f32311c161504e0a2be0f3c8f7f29e41923e.tar.gz -o /notebook_env/visionts.tar.gz

curl -L https://github.com/thuml/Large-Time-Series-Model/archive/fee65cb8fbd0a1474a23829d68e9e2ed23ff16ab.tar.gz -o /notebook_env/timer.tar.gz
```

After downloading these repos, intialize a virtual environment for each model:
```sh
MODEL_NAME = #change this accordingly
mkdir -p "/venvs/${MODEL_NAME}_eval_env"
python -m venv "/venvs/${MODEL_NAME}_eval_env"
source "/venvs/${MODEL_NAME}_eval_env/bin/activate"
```

Then follow the intallation instructions within each of the repository for enviornment setup.

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
