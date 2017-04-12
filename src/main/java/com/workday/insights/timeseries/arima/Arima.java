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
import java.util.List;

/**
 * ARIMA implementation
 */
public final class Arima {

    private Arima() {
    } // pure static class

    /**
     * Raw-level ARIMA forecasting function.
     *
     * @param data UNMODIFIED, list of double numbers representing time-series with constant time-gap
     * @param forecastSize integer representing how many data points AFTER the data series to be
     *        forecasted
     * @param p ARIMA parameter, the order (number of time lags) of the autoregressive model
     * @param d ARIMA parameter, the degree of differencing
     * @param q ARIMA parameter, the order of the moving-average model
     * @param P ARIMA parameter, autoregressive term for the seasonal part
     * @param D ARIMA parameter, differencing term for the seasonal part
     * @param Q ARIMA parameter, moving average term for the seasonal part
     * @param m ARIMA parameter, the number of periods in each season
     * @return a ForecastResult object, which contains the forecasted values and/or error message(s)
     */
    public static ForecastResult forecast_arima(final double[] data, final int forecastSize,
                                                int p, int d, int q, int P, int D, int Q, int m) {

        try {

            final ArimaParams paramsForecast = new ArimaParams(p, d, q, P, D, Q, m);
            final ArimaParams paramsXValidation = new ArimaParams(p, d, q, P, D, Q, m);
            // estimate ARIMA model parameters for forecasting
            final ArimaModel fittedModel = ArimaSolver.estimateARIMA(
                paramsForecast, data, data.length, data.length + 1);

            // compute RMSE to be used in confidence interval computation
            final double rmseValidation = ArimaSolver.computeRMSEValidation(
                data, ForecastUtil.testSetPercentage, paramsXValidation);
            fittedModel.setRMSE(rmseValidation);
            final ForecastResult forecastResult = fittedModel.forecast(forecastSize);

            // populate confidence interval
            forecastResult.setSigma2AndPredicationInterval(fittedModel.getParams());

            // add logging messages
            forecastResult.log("{" +
                               "\"Best ModelInterface Param\" : \"" + fittedModel.getParams().summary() + "\"," +
                               "\"Forecast Size\" : \"" + forecastSize + "\"," +
                               "\"Input Size\" : \"" + data.length + "\"" +
                               "}");

            // successfully built ARIMA model and its forecast
            return forecastResult;

        } catch (final Exception ex) {
            // failed to build ARIMA model
            throw new RuntimeException("Failed to build ARIMA forecast: " + ex.getMessage());
        }
    }
}
