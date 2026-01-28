---
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - time series
  - forecasting
  - pretrained models
  - foundation models
  - time series foundation models
  - time-series
---

# Chronos-Bolt⚡ (Base)

🚀 **Update Feb 14, 2025**: Chronos-Bolt models are now available on Amazon SageMaker JumpStart! Check out the [tutorial notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb) to learn how to deploy Chronos endpoints for production use in a few lines of code.

Chronos-Bolt is a family of pretrained time series forecasting models which can be used for zero-shot forecasting. It is based on the [T5 encoder-decoder architecture](https://arxiv.org/abs/1910.10683) and has been trained on nearly 100 billion time series observations. It chunks the historical time series context into patches of multiple observations, which are then input into the encoder. The decoder then uses these representations to directly generate quantile forecasts across multiple future steps—a method known as direct multi-step forecasting. Chronos-Bolt models are **more accurate**, up to **250 times faster** and **20 times more memory-efficient** than the [original Chronos](https://arxiv.org/abs/2403.07815) models of the same size.

## Performance

The following plot compares the inference time of Chronos-Bolt against the original Chronos models for forecasting 1024 time series with a context length of 512 observations and a prediction horizon of 64 steps.

<center>
<img src="https://autogluon.s3.amazonaws.com/images/chronos_bolt_speed.svg" width="50%"/>
</center>

Chronos-Bolt models are not only significantly faster but also more accurate than the original Chronos models. The following plot reports the probabilistic and point forecasting performance of Chronos-Bolt in terms of the [Weighted Quantile Loss (WQL)](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html#autogluon.timeseries.metrics.WQL) and the [Mean Absolute Scaled Error (MASE)](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html#autogluon.timeseries.metrics.MASE), respectively, aggregated over 27 datasets (see the [Chronos paper](https://arxiv.org/abs/2403.07815) for details on this benchmark). Remarkably, despite having no prior exposure to these datasets during training, the zero-shot Chronos-Bolt models outperform commonly used statistical models and deep learning models that have been trained on these datasets (highlighted by *). Furthermore, they also perform better than other FMs, denoted by a +, which indicates that these models were pretrained on certain datasets in our benchmark and are not entirely zero-shot. Notably, Chronos-Bolt (Base) also surpasses the original Chronos (Large) model in terms of the forecasting accuracy while being over 600 times faster.

<center>
<img src="https://autogluon.s3.amazonaws.com/images/chronos_bolt_accuracy.svg" width="80%"/>
</center>

Chronos-Bolt models are available in the following sizes.


<div align="center">

| Model                                                                  | Parameters | Based on                                                               |
| ---------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| [**chronos-bolt-tiny**](https://huggingface.co/autogluon/chronos-bolt-tiny)   | 9M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-bolt-mini**](https://huggingface.co/autogluon/chronos-bolt-mini)   | 21M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-bolt-small**](https://huggingface.co/autogluon/chronos-bolt-small) | 48M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-bolt-base**](https://huggingface.co/autogluon/chronos-bolt-base)   | 205M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |

</div>


## Usage

### Zero-shot inference with Chronos-Bolt in AutoGluon

Install the required dependencies.
```
pip install autogluon
```
Forecast with the Chronos-Bolt model.
```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

df = TimeSeriesDataFrame("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

predictor = TimeSeriesPredictor(prediction_length=48).fit(
    df,
    hyperparameters={
        "Chronos": {"model_path": "autogluon/chronos-bolt-base"},
    },
)

predictions = predictor.predict(df)
```

For more advanced features such as **fine-tuning** and **forecasting with covariates**, check out [this tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html).

### Deploying a Chronos-Bolt endpoint to SageMaker
First, update the SageMaker SDK to make sure that all the latest models are available.
```
pip install -U sagemaker
```
Deploy an inference endpoint to SageMaker.
```python
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(
    model_id="autogluon-forecasting-chronos-bolt-base",
    instance_type="ml.c5.2xlarge",
)
predictor = model.deploy()
```
Now you can send time series data to the endpoint in JSON format.
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

payload = {
    "inputs": [
        {"target": df["#Passengers"].tolist()}
    ],
    "parameters": {
        "prediction_length": 12,
    }
}
forecast = predictor.predict(payload)["predictions"]
```
Chronos-Bolt models can be deployed to both CPU and GPU instances. These models also support **forecasting with covariates**. For more details about the endpoint API, check out the [example notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb).

## Citation

If you find Chronos or Chronos-Bolt models useful for your research, please consider citing the associated [paper](https://arxiv.org/abs/2403.07815):

```
@article{ansari2024chronos,
    title={Chronos: Learning the Language of Time Series},
    author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=gerNCVqqtR}
}
```

## License

This project is licensed under the Apache-2.0 License.
