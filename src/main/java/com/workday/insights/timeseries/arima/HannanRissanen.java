/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima;

import com.workday.insights.matrix.InsightsMatrix;
import com.workday.insights.matrix.InsightsVector;
import com.workday.insights.timeseries.arima.struct.ArimaParams;
import com.workday.insights.timeseries.arima.struct.BackShift;
import com.workday.insights.timeseries.timeseriesutil.ForecastUtil;

/**
 * Hannan-Rissanen algorithm for estimating ARMA parameters
 */
public final class HannanRissanen {

    private HannanRissanen() {
    }

    /**
     * Estimate ARMA(p,q) parameters, i.e. AR-parameters: \phi_1, ... , \phi_p
     *                                     MA-parameters: \theta_1, ... , \theta_q
     * Input data is assumed to be stationary, has zero-mean, aligned, and imputed
     *
     * @param data_orig original data
     * @param params ARIMA parameters
     * @param forecast_length forecast length
     * @param maxIteration maximum number of iteration
     */
    public static void estimateARMA(final double[] data_orig, final ArimaParams params,
        final int forecast_length, final int maxIteration) {
        final double[] data = new double[data_orig.length];
        final int total_length = data.length;
        System.arraycopy(data_orig, 0, data, 0, total_length);
        final int r = (params.getDegreeP() > params.getDegreeQ()) ?
            1 + params.getDegreeP() : 1 + params.getDegreeQ();
        final int length = total_length - forecast_length;
        final int size = length - r;
        if (length < 2 * r) {
            throw new RuntimeException("not enough data points: length=" + length + ", r=" + r);
        }

        // step 1: apply Yule-Walker method and estimate AR(r) model on input data
        final double[] errors = new double[length];
        final double[] yuleWalkerParams = applyYuleWalkerAndGetInitialErrors(data, r, length,
            errors);
        for (int j = 0; j < r; ++j) {
            errors[j] = 0;
        }

        // step 2: iterate Least-Square fitting until the parameters converge
        // instantiate Z-matrix
        final double[][] matrix = new double[params.getNumParamsP() + params.getNumParamsQ()][size];

        double bestRMSE = -1; // initial value
        int remainIteration = maxIteration;
        InsightsVector bestParams = null;
        while (--remainIteration >= 0) {
            final InsightsVector estimatedParams = iterationStep(params, data, errors, matrix, r,
                length,
                size);
            final InsightsVector originalParams = params.getParamsIntoVector();
            params.setParamsFromVector(estimatedParams);

            // forecast for validation data and compute RMSE
            final double[] forecasts = ArimaSolver.forecastARMA(params, data, length, data.length);
            final double anotherRMSE = ArimaSolver
                .computeRMSE(data, forecasts, length, 0, forecast_length);
            // update errors
            final double[] train_forecasts = ArimaSolver.forecastARMA(params, data, r, data.length);
            for (int j = 0; j < size; ++j) {
                errors[j + r] = data[j + r] - train_forecasts[j];
            }
            if (bestRMSE < 0 || anotherRMSE < bestRMSE) {
                bestParams = estimatedParams;
                bestRMSE = anotherRMSE;
            }
        }
        params.setParamsFromVector(bestParams);
    }

    private static double[] applyYuleWalkerAndGetInitialErrors(final double[] data, final int r,
        final int length, final double[] errors) {
        final double[] yuleWalker = YuleWalker.fit(data, r);
        final BackShift bsYuleWalker = new BackShift(r, true);
        bsYuleWalker.initializeParams(false);
        // return array from YuleWalker is an array of size r whose
        // 0-th index element is lag 1 coefficient etc
        // hence shifting lag index by one and copy over to BackShift operator
        for (int j = 0; j < r; ++j) {
            bsYuleWalker.setParam(j + 1, yuleWalker[j]);
        }
        int m = 0;
        // populate error array
        while (m < r) {
            errors[m++] = 0;
        } // initial r-elements are set to zero
        while (m < length) {
            // from then on, initial estimate of error terms are
            // Z_t = X_t - \phi_1 X_{t-1} - \cdots - \phi_r X_{t-r}
            errors[m] = data[m] - bsYuleWalker.getLinearCombinationFrom(data, m);
            ++m;
        }
        return yuleWalker;
    }

    private static InsightsVector iterationStep(
        final ArimaParams params,
        final double[] data, final double[] errors,
        final double[][] matrix, final int r, final int length, final int size) {

        int rowIdx = 0;
        // copy over shifted timeseries data into matrix
        final int[] offsetsAR = params.getOffsetsAR();
        for (int pIdx : offsetsAR) {
            System.arraycopy(data, r - pIdx, matrix[rowIdx], 0, size);
            ++rowIdx;
        }
        // copy over shifted errors into matrix
        final int[] offsetsMA = params.getOffsetsMA();
        for (int qIdx : offsetsMA) {
            System.arraycopy(errors, r - qIdx, matrix[rowIdx], 0, size);
            ++rowIdx;
        }

        // instantiate matrix to perform least squares algorithm
        final InsightsMatrix zt = new InsightsMatrix(matrix, false);

        // instantiate target vector
        final double[] vector = new double[size];
        System.arraycopy(data, r, vector, 0, size);
        final InsightsVector x = new InsightsVector(vector, false);

        // obtain least squares solution
        final InsightsVector ztx = zt.timesVector(x);
        final InsightsMatrix ztz = zt.computeAAT();
        final InsightsVector estimatedVector = ztz
            .solveSPDIntoVector(ztx, ForecastUtil.maxConditionNumber);

        return estimatedVector;
    }
}
