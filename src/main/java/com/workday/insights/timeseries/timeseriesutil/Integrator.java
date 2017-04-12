/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.timeseriesutil;

/**
 * Pure Helper Class
 */
public class Integrator {

    private Integrator() {
    } // this is pure helper class

    /**
     *
     * @param src source array of data
     * @param dst destination array to store data
     * @param initial initial conditions
     * @param d length of initial conditions
     */
    public static void differentiate(final double[] src, final double[] dst, final double[] initial,
        final int d) {
        if (initial == null || initial.length != d || d <= 0) {
            throw new RuntimeException("invalid initial size=" + initial.length + ", d=" + d);
        }
        if (src == null || src.length <= d) {
            throw new RuntimeException("insufficient source size=" + src.length + ", d=" + d);
        }
        if (dst == null || dst.length != src.length - d) {
            throw new RuntimeException(
                "invalid destination size=" + dst.length + ", src=" + src.length + ", d=" + d);
        }

        // copy over initial conditions
        System.arraycopy(src, 0, initial, 0, d);
        // now differentiate source into destination
        final int src_len = src.length;
        for (int j = d, k = 0; j < src_len; ++j, ++k) {
            dst[k] = src[j] - src[k];
        }
    }

    /**
     *
     * @param src source array of data
     * @param dst destination array to store data
     * @param initial initial conditions
     * @param d length of initial conditions
     */
    public static void integrate(final double[] src, final double[] dst, final double[] initial,
        final int d) {
        if (initial == null || initial.length != d || d <= 0) {
            throw new RuntimeException("invalid initial size=" + initial.length + ", d=" + d);
        }
        if (dst == null || dst.length <= d) {
            throw new RuntimeException("insufficient destination size=" + dst.length + ", d=" + d);
        }
        if (src == null || src.length != dst.length - d) {
            throw new RuntimeException(
                "invalid source size=" + src.length + ", dst=" + dst.length + ", d=" + d);
        }

        // copy over initial conditions
        System.arraycopy(initial, 0, dst, 0, d);
        // now integrate source into destination
        final int src_len = src.length;
        for (int j = d, k = 0; k < src_len; ++j, ++k) {
            dst[j] = dst[k] + src[k];
        }
    }

    /**
     * Shifting the input data
     *
     * @param inputData MODIFIED. input data
     * @param shiftAmount shift amount
     */
    public static void shift(double[] inputData, final double shiftAmount) {
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] += shiftAmount;
        }
    }

    /**
     * Compute the mean of input data
     *
     * @param data input data
     * @return mean
     */
    public static double computeMean(final double[] data) {
        final int length = data.length;
        if (length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (int i = 0; i < length; ++i) {
            sum += data[i];
        }
        return sum / length;
    }

    /**
     * Compute the variance of input data
     *
     * @param data input data
     * @return variance
     */
    public static double computeVariance(final double[] data) {
        double variance = 0.0;
        double mean = computeMean(data);
        for (int i = 0; i < data.length; i++) {
            final double diff = data[i] - mean;
            variance += diff * diff;
        }
        return variance / (double) (data.length - 1);
    }
}
