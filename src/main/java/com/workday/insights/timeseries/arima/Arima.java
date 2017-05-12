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
     * @param params ARIMA parameters
     * @return a ForecastResult object, which contains the forecasted values and/or error message(s)
     */
    public static ForecastResult forecast_arima(final double[] data, final int forecastSize, ArimaParams params) {

        try {
            final int p = params.p;
            final int d = params.d;
            final int q = params.q;
            final int P = params.P;
            final int D = params.D;
            final int Q = params.Q;
            final int m = params.m;
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
