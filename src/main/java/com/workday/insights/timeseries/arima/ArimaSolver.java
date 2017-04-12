/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima;

import com.workday.insights.timeseries.arima.struct.ArimaModel;
import com.workday.insights.timeseries.arima.struct.ArimaParams;
import com.workday.insights.timeseries.arima.struct.ForecastResult;
import com.workday.insights.timeseries.timeseriesutil.ForecastUtil;
import com.workday.insights.timeseries.timeseriesutil.Integrator;

public final class ArimaSolver {

    private static final int maxIterationForHannanRissanen = 5;

    private ArimaSolver() {
    } // pure static helper class

    /**
     * Forecast ARMA
     *
     * @param params MODIFIED. ARIMA parameters
     * @param dataStationary UNMODIFIED. the time series AFTER differencing / centering
     * @param startIndex the index where the forecast starts. startIndex &le; data.length
     * @param endIndex the index where the forecast stops (exclusive). startIndex  endIndex
     * @return forecast ARMA data point
     */
    public static double[] forecastARMA(final ArimaParams params, final double[] dataStationary,
        final int startIndex, final int endIndex) {

        final int train_len = startIndex;
        final int total_len = endIndex;
        final double[] errors = new double[total_len];
        final double[] data = new double[total_len];
        System.arraycopy(dataStationary, 0, data, 0, train_len);
        final int forecast_len = endIndex - startIndex;
        final double[] forecasts = new double[forecast_len];
        final int _dp = params.getDegreeP();
        final int _dq = params.getDegreeQ();
        final int start_idx = (_dp > _dq) ? _dp : _dq;

        for (int j = 0; j < start_idx; ++j) {
            errors[j] = 0;
        }
        // populate errors and forecasts
        for (int j = start_idx; j < train_len; ++j) {
            final double forecast = params.forecastOnePointARMA(data, errors, j);
            final double error = data[j] - forecast;
            errors[j] = error;
        }
        // now we can forecast
        for (int j = train_len; j < total_len; ++j) {
            final double forecast = params.forecastOnePointARMA(data, errors, j);
            data[j] = forecast;
            errors[j] = 0;
            forecasts[j - train_len] = forecast;
        }
        // return forecasted values
        return forecasts;
    }

    /**
     * Produce forecast result based on input ARIMA parameters and forecast length.
     *
     * @param params UNMODIFIED. ARIMA parameters
     * @param data UNMODIFIED. the original time series before differencing / centering
     * @param forecastStartIndex the index where the forecast starts. startIndex &le; data.length
     * @param forecastEndIndex the index where the forecast stops (exclusive). startIndex &lt; endIndex
     * @return forecast result
     */
    public static ForecastResult forecastARIMA(final ArimaParams params, final double[] data,
        final int forecastStartIndex, final int forecastEndIndex) {

        if (!checkARIMADataLength(params, data, forecastStartIndex, forecastEndIndex)) {
            final int initialConditionSize = params.d + params.D * params.m;
            throw new RuntimeException(
                "not enough data for ARIMA. needed at least " + initialConditionSize +
                    ", have " + data.length + ", startIndex=" + forecastStartIndex + ", endIndex="
                    + forecastEndIndex);
        }

        final int forecast_length = forecastEndIndex - forecastStartIndex;
        final double[] forecast = new double[forecast_length];
        final double[] data_train = new double[forecastStartIndex];
        System.arraycopy(data, 0, data_train, 0, forecastStartIndex);

        //=======================================
        // DIFFERENTIATE
        final boolean hasSeasonalI = params.D > 0 && params.m > 0;
        final boolean hasNonSeasonalI = params.d > 0;
        double[] data_stationary = differentiate(params, data_train, hasSeasonalI,
            hasNonSeasonalI);  // currently un-centered
        // END OF DIFFERENTIATE
        //==========================================

        //=========== CENTERING ====================
        final double mean_stationary = Integrator.computeMean(data_stationary);
        Integrator.shift(data_stationary, (-1) * mean_stationary);
        final double dataVariance = Integrator.computeVariance(data_stationary);
        //==========================================

        //==========================================
        // FORECAST
        final double[] forecast_stationary = forecastARMA(params, data_stationary,
            data_stationary.length,
            data_stationary.length + forecast_length);

        final double[] data_forecast_stationary = new double[data_stationary.length
            + forecast_length];

        System.arraycopy(data_stationary, 0, data_forecast_stationary, 0, data_stationary.length);
        System.arraycopy(forecast_stationary, 0, data_forecast_stationary, data_stationary.length,
            forecast_stationary.length);
        // END OF FORECAST
        //==========================================

        //=========== UN-CENTERING =================
        Integrator.shift(data_forecast_stationary, mean_stationary);
        //==========================================

        //===========================================
        // INTEGRATE
        double[] forecast_merged = integrate(params, data_forecast_stationary, hasSeasonalI,
            hasNonSeasonalI);
        // END OF INTEGRATE
        //===========================================
        System.arraycopy(forecast_merged, forecastStartIndex, forecast, 0, forecast_length);

        return new ForecastResult(forecast, dataVariance);
    }

    /**
     * Creates the fitted ARIMA model based on the ARIMA parameters.
     *
     * @param params MODIFIED. ARIMA parameters
     * @param data UNMODIFIED. the original time series before differencing / centering
     * @param forecastStartIndex the index where the forecast starts. startIndex &le; data.length
     * @param forecastEndIndex the index where the forecast stops (exclusive). startIndex &lt; endIndex
     * @return fitted ARIMA model
     */
    public static ArimaModel estimateARIMA(final ArimaParams params, final double[] data,
        final int forecastStartIndex, final int forecastEndIndex) {

        if (!checkARIMADataLength(params, data, forecastStartIndex, forecastEndIndex)) {
            final int initialConditionSize = params.d + params.D * params.m;
            throw new RuntimeException(
                "not enough data for ARIMA. needed at least " + initialConditionSize +
                    ", have " + data.length + ", startIndex=" + forecastStartIndex + ", endIndex="
                    + forecastEndIndex);
        }

        final int forecast_length = forecastEndIndex - forecastStartIndex;
        final double[] data_train = new double[forecastStartIndex];
        System.arraycopy(data, 0, data_train, 0, forecastStartIndex);

        //=======================================
        // DIFFERENTIATE
        final boolean hasSeasonalI = params.D > 0 && params.m > 0;
        final boolean hasNonSeasonalI = params.d > 0;
        double[] data_stationary = differentiate(params, data_train, hasSeasonalI,
            hasNonSeasonalI);  // currently un-centered
        // END OF DIFFERENTIATE
        //==========================================

        //=========== CENTERING ====================
        final double mean_stationary = Integrator.computeMean(data_stationary);
        Integrator.shift(data_stationary, (-1) * mean_stationary);
        //==========================================

        //==========================================
        // FORECAST
        HannanRissanen
            .estimateARMA(data_stationary, params, forecast_length,
                maxIterationForHannanRissanen);

        return new ArimaModel(params, data, forecastStartIndex);
    }

    /**
     * Differentiate procedures for forecast and estimate ARIMA.
     *
     * @param params ARIMA parameters
     * @param trainingData training data
     * @param hasSeasonalI has seasonal I or not based on the parameter
     * @param hasNonSeasonalI has NonseasonalI or not based on the parameter
     * @return stationary data
     */
    private static double[] differentiate(ArimaParams params, double[] trainingData,
        boolean hasSeasonalI, boolean hasNonSeasonalI) {
        double[] dataStationary;  // currently un-centered
        if (hasSeasonalI && hasNonSeasonalI) {
            params.differentiateSeasonal(trainingData);
            params.differentiateNonSeasonal(params.getLastDifferenceSeasonal());
            dataStationary = params.getLastDifferenceNonSeasonal();
        } else if (hasSeasonalI) {
            params.differentiateSeasonal(trainingData);
            dataStationary = params.getLastDifferenceSeasonal();
        } else if (hasNonSeasonalI) {
            params.differentiateNonSeasonal(trainingData);
            dataStationary = params.getLastDifferenceNonSeasonal();
        } else {
            dataStationary = new double[trainingData.length];
            System.arraycopy(trainingData, 0, dataStationary, 0, trainingData.length);
        }

        return dataStationary;
    }

    /**
     * Differentiate procedures for forecast and estimate ARIMA.
     *
     * @param params ARIMA parameters
     * @param dataForecastStationary stationary forecast data
     * @param hasSeasonalI has seasonal I or not based on the parameter
     * @param hasNonSeasonalI has NonseasonalI or not based on the parameter
     * @return merged forecast data
     */
    private static double[] integrate(ArimaParams params, double[] dataForecastStationary,
        boolean hasSeasonalI, boolean hasNonSeasonalI) {
        double[] forecast_merged;
        if (hasSeasonalI && hasNonSeasonalI) {
            params.integrateSeasonal(dataForecastStationary);
            params.integrateNonSeasonal(params.getLastIntegrateSeasonal());
            forecast_merged = params.getLastIntegrateNonSeasonal();
        } else if (hasSeasonalI) {
            params.integrateSeasonal(dataForecastStationary);
            forecast_merged = params.getLastIntegrateSeasonal();
        } else if (hasNonSeasonalI) {
            params.integrateNonSeasonal(dataForecastStationary);
            forecast_merged = params.getLastIntegrateNonSeasonal();
        } else {
            forecast_merged = new double[dataForecastStationary.length];
            System.arraycopy(dataForecastStationary, 0, forecast_merged, 0,
                dataForecastStationary.length);
        }

        return forecast_merged;
    }

    /**
     * Computes the Root Mean-Squared Error given a time series (with forecast) and true values
     *
     * @param left time series being evaluated
     * @param right true values
     * @param startIndex the index which to start evaluation
     * @param endIndex the index which to end evaluation
     * @param leftIndexOffset the number of elements from @param startIndex to the index which
     * evaluation begins
     * @return Root Mean-Squared Error
     */
    public static double computeRMSE(final double[] left, final double[] right,
        final int leftIndexOffset,
        final int startIndex, final int endIndex) {

        final int len_left = left.length;
        final int len_right = right.length;
        if (startIndex >= endIndex || startIndex < 0 || len_right < endIndex ||
            len_left + leftIndexOffset < 0 || len_left + leftIndexOffset < endIndex) {
            throw new RuntimeException(
                "invalid arguments: startIndex=" + startIndex + ", endIndex=" + endIndex +
                    ", len_left=" + len_left + ", len_right=" + len_right + ", leftOffset="
                    + leftIndexOffset);
        }
        double square_sum = 0.0;
        for (int i = startIndex; i < endIndex; ++i) {
            final double error = left[i + leftIndexOffset] - right[i];
            square_sum += error * error;
        }
        return Math.sqrt(square_sum / (double) (endIndex - startIndex));
    }

    /**
     * Performs validation using Root Mean-Squared Error given a time series (with forecast) and
     * true values
     *
     * @param data UNMODIFIED. time series data being evaluated
     * @param testDataPercentage percentage of data to be used to evaluate as test set
     * @param params MODIFIED. parameter of the ARIMA model
     * @return a Root Mean-Squared Error computed from the forecast and true data
     */
    public static double computeRMSEValidation(final double[] data,
        final double testDataPercentage, ArimaParams params) {

        int testDataLength = (int) (data.length * testDataPercentage);
        int trainingDataEndIndex = data.length - testDataLength;

        final ArimaModel result = estimateARIMA(params, data, trainingDataEndIndex,
            data.length);

        final double[] forecast = result.forecast(testDataLength).getForecast();

        return computeRMSE(data, forecast, trainingDataEndIndex, 0, forecast.length);
    }

    /**
     * Set Sigma2(RMSE) and Predication Interval for forecast result.
     *
     * @param params ARIMA parameters
     * @param forecastResult MODIFIED. forecast result
     * @param forecastSize size of forecast
     * @return max normalized variance
     */
    public static double setSigma2AndPredicationInterval(final ArimaParams params,
        final ForecastResult forecastResult, final int forecastSize) {

        final double[] coeffs_AR = params.getCurrentARCoefficients();
        final double[] coeffs_MA = params.getCurrentMACoefficients();
        return forecastResult
            .setConfInterval(ForecastUtil.confidence_constant_95pct,
                ForecastUtil.getCumulativeSumOfCoeff(
                    ForecastUtil.ARMAtoMA(coeffs_AR, coeffs_MA, forecastSize)));
    }

    /**
     * Input checker
     *
     * @param params ARIMA parameter
     * @param data original data
     * @param startIndex start index of ARIMA operation
     * @param endIndex end index of ARIMA operation
     * @return whether the inputs are valid
     */
    private static boolean checkARIMADataLength(ArimaParams params, double[] data, int startIndex,
        int endIndex) {
        boolean result = true;

        final int initialConditionSize = params.d + params.D * params.m;

        if (data.length < initialConditionSize || startIndex < initialConditionSize
            || endIndex <= startIndex) {
            result = false;
        }

        return result;
    }
}
