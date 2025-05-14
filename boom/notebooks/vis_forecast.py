import matplotlib.pyplot as plt
import numpy as np
from gluonts.model.forecast import QuantileForecast, SampleForecast


def visualize(
    forecasts, dataset, process_inputs=None, xlim=None, ylim=None, num_pictures=[0], verbose=True, fig_size=(10, 2)
):
    # ==============================================================================
    # forecasts: list of forecast (SampleForecast, QuantileForecast, or list like iterables)
    #     expected shape of forecast[i] to be (num_samples, num_steps, num_variate) for gluonts sample object or (num_variate, num_steps) for numpy array
    # process_inputs: list of processed input samples
    # dataset: dataset object
    # xlim: tuple of x-axis limits, support only 1 element in the tuple as the left bound
    # ylim: tuple of y-axis limits, support only 1 element in the tuple as the bottom bound
    # num_pictures: list of indices of pictures to plot
    # ==============================================================================

    mutlivariate_switch = False
    num_variate = 0
    upper_bound = []
    lower_bound = []
    mean_forecast = []
    if isinstance(forecasts[0], QuantileForecast):
        if verbose:
            print("QuantileForecast object detected")
        assert (
            len(forecasts[0].forecast_array.shape) == 2 or len(forecasts[0].forecast_array.shape) == 3
        ), "Samples should be in shape of (num_samples, num_steps, num_variate)"
        if len(forecasts[0].forecast_array.shape) == 3:
            mutlivariate_switch = True
            num_variate = forecasts[0].forecast_array.shape[2]
        for forecast in forecasts:
            upper_bound.append(forecast.quantile(0.9))
            lower_bound.append(forecast.quantile(0.1))
            try:
                mean_forecast.append(forecast.mean)
            except:
                if verbose:
                    print("No mean in forecast, using median instead")
                mean_forecast.append(forecast.quantile(0.5))
    elif isinstance(forecasts[0], SampleForecast):
        if verbose:
            print("SampleForecast object detected")
        assert (
            len(forecasts[0].samples.shape) == 2 or len(forecasts[0].samples.shape) == 3
        ), "Samples should be in shape of (num_samples, num_steps, num_variate)"
        if len(forecasts[0].samples.shape) == 3:
            mutlivariate_switch = True
            num_variate = forecasts[0].samples.shape[2]
        for forecast in forecasts:
            upper_bound.append(forecast.samples.max(axis=0))
            lower_bound.append(forecast.samples.min(axis=0))
            try:
                mean_forecast.append(forecast.mean)
            except:
                mean_forecast.append(forecast.samples.mean(axis=0))
    else:
        if verbose:
            print("Automatically recognizing forecasts as list with 1 samples")
        assert (
            len(forecasts[0].shape) == 1 or len(forecasts[0].shape) == 2
        ), "Samples should be in shape of (num_variate, num_steps)"
        if len(forecasts[0].shape) == 2:
            mutlivariate_switch = True
            num_variate = forecasts[0].shape[0]
            for forecast in forecasts:
                upper_bound.append(forecast.transpose(1, 0))
                lower_bound.append(forecast.transpose(1, 0))
                mean_forecast.append(forecast.transpose(1, 0))
        else:
            upper_bound = forecasts
            lower_bound = forecasts
            mean_forecast = forecasts

    fig, axes = plt.subplots(len(num_pictures), 1, figsize=(fig_size[0], fig_size[1] * len(num_pictures)))
    if len(num_pictures) == 1:
        axes = [axes]
    test_inputs = dataset.test_data.input
    test_labels = dataset.test_data.label

    if verbose:
        print("Number of variate is:", num_variate)
        if mutlivariate_switch:
            print("Total number of pictures is:", num_variate * len(dataset.test_data))
        else:
            print("Total number of pictures is:", len(dataset.test_data))

    for idx, num_picture in enumerate(num_pictures):
        test_inputs_iter = iter(test_inputs)
        test_labels_iter = iter(test_labels)
        if not mutlivariate_switch:
            # print(next(iter(test_inputs))["target"].shape)
            # print(next(iter(test_labels))["target"].shape)
            # print(forecast[0].samples.shape)

            for i in range(num_picture + 1):
                test_input = next(test_inputs_iter)["target"]
                test_label = next(test_labels_iter)["target"]

            if process_inputs:
                process_input = process_inputs[num_picture].reshape(-1)
            else:
                process_input = test_input
            test_input = test_input.reshape(-1)
            test_label = test_label.reshape(-1)
            forecast_mean = mean_forecast[num_picture].reshape(-1)
            forecast_upper = upper_bound[num_picture].reshape(-1)
            forecast_lower = lower_bound[num_picture].reshape(-1)
        else:
            variate_idx = num_picture // num_variate
            variate_num = num_picture % num_variate
            for i in range(variate_idx + 1):
                test_input = next(test_inputs_iter)["target"]
                test_label = next(test_labels_iter)["target"]
            if num_variate == 1:
                test_input = test_input
                test_label = test_label
            else:
                test_input = test_input[variate_num, :].reshape(-1)
                test_label = test_label[variate_num, :].reshape(-1)
            if process_inputs:
                process_input = process_inputs[num_picture][variate_num, :].reshape(-1)
            else:
                process_input = test_input
            forecast_mean = mean_forecast[num_picture][:, variate_num].reshape(-1)
            forecast_upper = upper_bound[num_picture][:, variate_num].reshape(-1)
            forecast_lower = lower_bound[num_picture][:, variate_num].reshape(-1)

        if verbose:
            print("shape of raw input is:", test_input.shape)
            print("shape of processed input is:", process_input.shape)
            print("shape of label is:", test_label.shape)
            print("shape of forecast is:", forecast_mean.shape)
            print("shape of upper bound is:", forecast_upper.shape)
            print("shape of lower bound is:", forecast_lower.shape)

        truncate_len = process_input.shape[0]
        x1_all = np.arange(truncate_len)
        x1 = np.arange(test_input.shape[0]) + truncate_len - test_input.shape[0]
        x2 = np.arange(test_label.shape[0]) + truncate_len

        ax = axes[idx]  # Select the correct subplot

        ax.plot(x1, test_input, label="Original Input")
        ax.plot(x2, test_label, label="Test Label")
        ax.plot(x2, forecast_mean, label="Forecast", linestyle="--")
        ax.plot(x1_all, process_input, label="Processed Input")
        ax.fill_between(x2, forecast_upper, forecast_lower, color="gray", alpha=0.5)

        if xlim is not None:
            if len(xlim) == 1:
                ax.set_xlim(left=xlim[0])
            else:
                ax.set_xlim(xlim)
        if ylim is not None:
            if len(ylim) == 1:
                ax.set_ylim(bottom=ylim[0])
            else:
                ax.set_ylim(ylim)

        ax.legend()
        ax.set_xticks([])

    plt.tight_layout()
    plt.show()
