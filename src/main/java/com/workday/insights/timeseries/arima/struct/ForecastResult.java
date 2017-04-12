/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima.struct;

import com.workday.insights.timeseries.arima.ArimaSolver;

/**
 * ARIMA Forecast Result
 */
public class ForecastResult {

    private final double[] forecast;
    private final double[] forecastUpperConf;
    private final double[] forecastLowerConf;
    private final double dataVariance;
    private final StringBuilder log;
    private double modelRMSE;
    private double maxNormalizedVariance;

    /**
     * Constructor for ForecastResult
     *
     * @param pForecast forecast data
     * @param pDataVariance data variance of the original data
     */
    public ForecastResult(final double[] pForecast, final double pDataVariance) {

        this.forecast = pForecast;

        this.forecastUpperConf = new double[pForecast.length];
        System.arraycopy(pForecast, 0, forecastUpperConf, 0, pForecast.length);

        this.forecastLowerConf = new double[pForecast.length];
        System.arraycopy(pForecast, 0, forecastLowerConf, 0, pForecast.length);

        this.dataVariance = pDataVariance;

        this.modelRMSE = -1;
        this.maxNormalizedVariance = -1;

        this.log = new StringBuilder();
    }

    /**
     * Compute normalized variance
     *
     * @param v variance
     * @return Normalized variance
     */
    private double getNormalizedVariance(final double v) {
        if (v < -0.5 || dataVariance < -0.5) {
            return -1;
        } else if (dataVariance < 0.0000001) {
            return v;
        } else {
            return Math.abs(v / dataVariance);
        }
    }

    /**
     * Getter for Root Mean-Squared Error
     *
     * @return Root Mean-Squared Error
     */
    public double getRMSE() {
        return this.modelRMSE;
    }

    /**
     * Setter for Root Mean-Squared Error
     *
     * @param rmse Root Mean-Squared Error
     */
    void setRMSE(double rmse) {
        this.modelRMSE = rmse;
    }

    /**
     * Getter for Max Normalized Variance
     *
     * @return Max Normalized Variance
     */
    public double getMaxNormalizedVariance() {
        return maxNormalizedVariance;
    }

    /**
     * Compute and set confidence intervals
     *
     * @param constant confidence interval constant
     * @param cumulativeSumOfMA cumulative sum of MA coefficients
     * @return Max Normalized Variance
     */
    public double setConfInterval(final double constant, final double[] cumulativeSumOfMA) {
        double maxNormalizedVariance = -1.0;
        double bound = 0;
        for (int i = 0; i < forecast.length; i++) {
            bound = constant * modelRMSE * cumulativeSumOfMA[i];
            this.forecastUpperConf[i] = this.forecast[i] + bound;
            this.forecastLowerConf[i] = this.forecast[i] - bound;
            final double normalizedVariance = getNormalizedVariance(Math.pow(bound, 2));
            if (normalizedVariance > maxNormalizedVariance) {
                maxNormalizedVariance = normalizedVariance;
            }
        }
        return maxNormalizedVariance;
    }

    /**
     * Compute and set Sigma2 and prediction confidence interval.
     *
     * @param params ARIMA parameters from the model
     */
    public void setSigma2AndPredicationInterval(ArimaParams params) {
        maxNormalizedVariance = ArimaSolver
            .setSigma2AndPredicationInterval(params, this, forecast.length);
    }

    /**
     * Getter for forecast data
     *
     * @return forecast data
     */
    public double[] getForecast() {
        return forecast;
    }

    /**
     * Getter for upper confidence bounds
     *
     * @return array of upper confidence bounds
     */
    public double[] getForecastUpperConf() {
        return forecastUpperConf;
    }

    /**
     * Getter for lower confidence bounds
     *
     * @return array of lower confidence bounds
     */
    public double[] getForecastLowerConf() {
        return forecastLowerConf;
    }

    /**
     * Append message to log of forecast result
     *
     * @param message string message
     */
    public void log(String message) {
        this.log.append(message + "\n");
    }

    /**
     * Getter for log of the forecast result
     *
     * @return full log of the forecast result
     */
    public String getLog() {
        return log.toString();
    }
}
