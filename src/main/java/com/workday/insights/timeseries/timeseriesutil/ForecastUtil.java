/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.timeseriesutil;

import com.workday.insights.matrix.InsightsMatrix;

/**
 * Time series forecasting Utilities
 */
public final class ForecastUtil {


    public static final double testSetPercentage = 0.15;
    public static final double maxConditionNumber = 100;
    public static final double confidence_constant_95pct = 1.959963984540054;

    private ForecastUtil() {
    }

    /**
     * Instantiates Toeplitz matrix from given input array
     *
     * @param input double array as input data
     * @return a Toeplitz InsightsMatrix
     */
    public static InsightsMatrix initToeplitz(double[] input) {
        int length = input.length;
        double toeplitz[][] = new double[length][length];

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                if (j > i) {
                    toeplitz[i][j] = input[j - i];
                } else if (j == i) {
                    toeplitz[i][j] = input[0];
                } else {
                    toeplitz[i][j] = input[i - j];
                }
            }
        }
        return new InsightsMatrix(toeplitz, false);
    }

    /**
     * Invert AR part of ARMA to obtain corresponding MA series
     *
     * @param ar AR portion of the ARMA
     * @param ma MA portion of the ARMA
     * @param lag_max maximum lag
     * @return MA series
     */
    public static double[] ARMAtoMA(final double[] ar, final double[] ma, final int lag_max) {
        final int p = ar.length;
        final int q = ma.length;
        final double[] psi = new double[lag_max];

        for (int i = 0; i < lag_max; i++) {
            double tmp = (i < q) ? ma[i] : 0.0;
            for (int j = 0; j < Math.min(i + 1, p); j++) {
                tmp += ar[j] * ((i - j - 1 >= 0) ? psi[i - j - 1] : 1.0);
            }
            psi[i] = tmp;
        }
        final double[] include_psi1 = new double[lag_max];
        include_psi1[0] = 1;
        for (int i = 1; i < lag_max; i++) {
            include_psi1[i] = psi[i - 1];
        }
        return include_psi1;
    }

    /**
     * Simple helper that returns cumulative sum of coefficients
     *
     * @param coeffs array of coefficients
     * @return array of cumulative sum of the coefficients
     */
    public static double[] getCumulativeSumOfCoeff(final double[] coeffs) {
        final int len = coeffs.length;
        final double[] cumulativeSquaredCoeffSumVector = new double[len];
        double cumulative = 0.0;
        for (int i = 0; i < len; i++) {
            cumulative += Math.pow(coeffs[i], 2);
            cumulativeSquaredCoeffSumVector[i] = Math.pow(cumulative, 0.5);
        }
        return cumulativeSquaredCoeffSumVector;
    }

}
