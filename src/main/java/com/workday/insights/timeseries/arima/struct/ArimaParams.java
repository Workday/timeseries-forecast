/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima.struct;

import com.workday.insights.matrix.InsightsVector;
import com.workday.insights.timeseries.timeseriesutil.Integrator;

/**
 * Simple wrapper for ARIMA parameters and fitted states
 */
public final class ArimaParams {

    public final int p;
    public final int d;
    public final int q;
    public final int P;
    public final int D;
    public final int Q;
    public final int m;
    // ARMA part
    private final BackShift _opAR;
    private final BackShift _opMA;
    private final int _dp;
    private final int _dq;
    private final int _np;
    private final int _nq;
    private final double[][] _init_seasonal;
    private final double[][] _diff_seasonal;
    private final double[][] _integrate_seasonal;
    private final double[][] _init_non_seasonal;
    private final double[][] _diff_non_seasonal;
    private final double[][] _integrate_non_seasonal;
    private int[] lagsAR = null;
    private double[] paramsAR = null;
    private int[] lagsMA = null;
    private double[] paramsMA = null;
    // I part
    private double _mean = 0.0;

    /**
     * Constructor for ArimaParams
     *
     * @param p ARIMA parameter, the order (number of time lags) of the autoregressive model
     * @param d ARIMA parameter, the degree of differencing
     * @param q ARIMA parameter, the order of the moving-average model
     * @param P ARIMA parameter, autoregressive term for the seasonal part
     * @param D ARIMA parameter, differencing term for the seasonal part
     * @param Q ARIMA parameter, moving average term for the seasonal part
     * @param m ARIMA parameter, the number of periods in each season
     */
    public ArimaParams(
        int p, int d, int q,
        int P, int D, int Q,
        int m) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.P = P;
        this.D = D;
        this.Q = Q;
        this.m = m;

        // dependent states
        this._opAR = getNewOperatorAR();
        this._opMA = getNewOperatorMA();
        _opAR.initializeParams(false);
        _opMA.initializeParams(false);
        this._dp = _opAR.getDegree();
        this._dq = _opMA.getDegree();
        this._np = _opAR.numParams();
        this._nq = _opMA.numParams();
        this._init_seasonal = (D > 0 && m > 0) ? new double[D][m] : null;
        this._init_non_seasonal = (d > 0) ? new double[d][1] : null;
        this._diff_seasonal = (D > 0 && m > 0) ? new double[D][] : null;
        this._diff_non_seasonal = (d > 0) ? new double[d][] : null;
        this._integrate_seasonal = (D > 0 && m > 0) ? new double[D][] : null;
        this._integrate_non_seasonal = (d > 0) ? new double[d][] : null;
    }

    /**
     * ARMA forecast of one data point.
     *
     * @param data input data
     * @param errors array of errors
     * @param index index
     * @return one data point
     */
    public double forecastOnePointARMA(final double[] data, final double[] errors,
        final int index) {
        final double estimateAR = _opAR.getLinearCombinationFrom(data, index);
        final double estimateMA = _opMA.getLinearCombinationFrom(errors, index);
        final double forecastValue = estimateAR + estimateMA;
        return forecastValue;
    }

    /**
     * Getter for the degree of parameter p
     *
     * @return degree of p
     */
    public int getDegreeP() {
        return _dp;
    }

    /**
     * Getter for the degree of parameter q
     *
     * @return degree of q
     */
    public int getDegreeQ() {
        return _dq;
    }

    /**
     * Getter for the number of parameters p
     * @return number of parameters p
     */
    public int getNumParamsP() {
        return _np;
    }

    /**
     * Getter for the number of parameters q
     *
     * @return number of parameters q
     */
    public int getNumParamsQ() {
        return _nq;
    }

    /**
     * Getter for the parameter offsets of AR
     *
     * @return parameter offsets of AR
     */
    public int[] getOffsetsAR() {
        return _opAR.paramOffsets();
    }

    /**
     * Getter for the parameter offsets of MA
     *
     * @return parameter offsets of MA
     */
    public int[] getOffsetsMA() {
        return _opMA.paramOffsets();
    }

    /**
     * Getter for the last integrated seasonal data
     *
     * @return integrated seasonal data
     */
    public double[] getLastIntegrateSeasonal() {
        return _integrate_seasonal[D - 1];
    }

    /**
     * Getter for the last integrated NON-seasonal data
     *
     * @return NON-integrated NON-seasonal data
     */
    public double[] getLastIntegrateNonSeasonal() {
        return _integrate_non_seasonal[d - 1];
    }

    /**
     * Getter for the last differentiated seasonal data
     *
     * @return differentiate seasonal data
     */
    public double[] getLastDifferenceSeasonal() {
        return _diff_seasonal[D - 1];
    }

    /**
     * Getter for the last differentiated NON-seasonal data
     *
     * @return differentiated NON-seasonal data
     */
    public double[] getLastDifferenceNonSeasonal() {
        return _diff_non_seasonal[d - 1];
    }

    /**
     * Summary of the parameters
     *
     * @return String of summary
     */
    public String summary() {
        return "ModelInterface ParamsInterface:" +
            ", p= " + p +
            ", d= " + d +
            ", q= " + q +
            ", P= " + P +
            ", D= " + D +
            ", Q= " + Q +
            ", m= " + m;
    }

    //==========================================================
    // MUTABLE STATES

    /**
     * Setting parameters from a Insight Vector
     *
     * It is assumed that the input vector has _np + _nq entries first _np entries are AR-parameters
     *      and the last _nq entries are MA-parameters
     *
     * @param paramVec a vector of parameters
     */
    public void setParamsFromVector(final InsightsVector paramVec) {
        int index = 0;
        final int[] offsetsAR = getOffsetsAR();
        final int[] offsetsMA = getOffsetsMA();
        for (int pIdx : offsetsAR) {
            _opAR.setParam(pIdx, paramVec.get(index++));
        }
        for (int qIdx : offsetsMA) {
            _opMA.setParam(qIdx, paramVec.get(index++));
        }
    }

    /**
     * Create a Insight Vector that contains the parameters.
     *
     * It is assumed that the input vector has _np + _nq entries first _np entries are AR-parameters
     *      and the last _nq entries are MA-parameters
     *
     * @return Insight Vector of parameters
     */
    public InsightsVector getParamsIntoVector() {
        int index = 0;
        final InsightsVector paramVec = new InsightsVector(_np + _nq, 0.0);
        final int[] offsetsAR = getOffsetsAR();
        final int[] offsetsMA = getOffsetsMA();
        for (int pIdx : offsetsAR) {
            paramVec.set(index++, _opAR.getParam(pIdx));
        }
        for (int qIdx : offsetsMA) {
            paramVec.set(index++, _opMA.getParam(qIdx));
        }
        return paramVec;
    }

    public BackShift getNewOperatorAR() {
        return mergeSeasonalWithNonSeasonal(p, P, m);
    }

    public BackShift getNewOperatorMA() {
        return mergeSeasonalWithNonSeasonal(q, Q, m);
    }

    public double[] getCurrentARCoefficients() {
        return _opAR.getCoefficientsFlattened();
    }

    public double[] getCurrentMACoefficients() {
        return _opMA.getCoefficientsFlattened();
    }

    private BackShift mergeSeasonalWithNonSeasonal(int nonSeasonalLag, int seasonalLag,
        int seasonalStep) {
        final BackShift nonSeasonal = new BackShift(nonSeasonalLag, true);
        final BackShift seasonal = new BackShift(seasonalLag * seasonalStep, false);
        for (int s = 1; s <= seasonalLag; ++s) {
            seasonal.setIndex(s * seasonalStep, true);
        }
        final BackShift merged = seasonal.apply(nonSeasonal);
        return merged;
    }

    //================================
    // Differentiation and Integration

    public void differentiateSeasonal(final double[] data) {
        double[] current = data;
        for (int j = 0; j < D; ++j) {
            final double[] next = new double[current.length - m];
            _diff_seasonal[j] = next;
            final double[] init = _init_seasonal[j];
            Integrator.differentiate(current, next, init, m);
            current = next;
        }
    }

    public void differentiateNonSeasonal(final double[] data) {
        double[] current = data;
        for (int j = 0; j < d; ++j) {
            final double[] next = new double[current.length - 1];
            _diff_non_seasonal[j] = next;
            final double[] init = _init_non_seasonal[j];
            Integrator.differentiate(current, next, init, 1);
            current = next;
        }
    }

    public void integrateSeasonal(final double[] data) {
        double[] current = data;
        for (int j = 0; j < D; ++j) {
            final double[] next = new double[current.length + m];
            _integrate_seasonal[j] = next;
            final double[] init = _init_seasonal[j];
            Integrator.integrate(current, next, init, m);
            current = next;
        }
    }

    public void integrateNonSeasonal(final double[] data) {
        double[] current = data;
        for (int j = 0; j < d; ++j) {
            final double[] next = new double[current.length + 1];
            _integrate_non_seasonal[j] = next;
            final double[] init = _init_non_seasonal[j];
            Integrator.integrate(current, next, init, 1);
            current = next;
        }
    }
}
