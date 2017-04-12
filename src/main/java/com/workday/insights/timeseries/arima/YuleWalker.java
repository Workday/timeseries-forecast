/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima;

import com.workday.insights.matrix.InsightsMatrix;
import com.workday.insights.matrix.InsightsVector;
import com.workday.insights.timeseries.timeseriesutil.ForecastUtil;
import java.util.Arrays;

/**
 * Yule-Walker algorithm implementation
 */
public final class YuleWalker {

    private YuleWalker() {
    }

    /**
     * Perform Yule-Walker algorithm to the given timeseries data
     *
     * @param data input data
     * @param p YuleWalker Parameter
     * @return array of Auto-Regressive parameter estimates. Index 0 contains coefficient of lag 1
     */
    public static double[] fit(final double[] data, final int p) {

        int length = data.length;
        if (length == 0 || p < 1) {
            throw new RuntimeException(
                "fitYuleWalker - Invalid Parameters" + "length=" + length + ", p = " + p);
        }

        double[] r = new double[p + 1];
        for (double aData : data) {
            r[0] += Math.pow(aData, 2);
        }
        r[0] /= length;

        for (int j = 1; j < p + 1; j++) {
            for (int i = 0; i < length - j; i++) {
                r[j] += data[i] * data[i + j];
            }
            r[j] /= (length);
        }

        final InsightsMatrix toeplitz = ForecastUtil.initToeplitz(Arrays.copyOfRange(r, 0, p));
        final InsightsVector rVector = new InsightsVector(Arrays.copyOfRange(r, 1, p + 1), false);

        return toeplitz.solveSPDIntoVector(rVector, ForecastUtil.maxConditionNumber).deepCopy();
    }
}
