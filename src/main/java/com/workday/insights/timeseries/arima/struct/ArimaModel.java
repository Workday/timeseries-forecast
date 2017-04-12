/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima.struct;

import com.workday.insights.timeseries.arima.ArimaSolver;

/**
 * ARIMA model
 */
public class ArimaModel {

    private final ArimaParams params;
    private final double[] data;
    private final int trainDataSize;
    private double rmse;

    /**
     * Constructor for ArimaModel
     *
     * @param params ARIMA parameter
     * @param data original data
     * @param trainDataSize size of train data
     */
    public ArimaModel(ArimaParams params, double[] data, int trainDataSize) {
        this.params = params;
        this.data = data;
        this.trainDataSize = trainDataSize;
    }

    /**
     * Getter for Root Mean-Squared Error.
     *
     * @return Root Mean-Squared Error for the ARIMA model
     */
    public double getRMSE() {
        return this.rmse;
    }

    /**
     * Setter for Root Mean-Squared Error
     *
     * @param rmse source Root Mean-Squared Error
     */
    public void setRMSE(final double rmse) {
        this.rmse = rmse;
    }

    /**
     * Getter for ARIMA parameters.
     *
     * @return ARIMA parameters for the model
     */
    public ArimaParams getParams() {
        return params;
    }

    /**
     * Forecast data base on training data and forecast size.
     *
     * @param forecastSize size of forecast
     * @return forecast result
     */
    public ForecastResult forecast(final int forecastSize) {
        ForecastResult forecastResult = ArimaSolver
            .forecastARIMA(params, data, trainDataSize, trainDataSize + forecastSize);
        forecastResult.setRMSE(this.rmse);

        return forecastResult;
    }

}
