/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima.struct;

/**
 * Helper class that implements polynomial of back-shift operator
 */
public final class BackShift {

    private final int _degree;  // maximum lag, e.g. AR(1) degree will be 1
    private final boolean[] _indices;
    private int[] _offsets = null;
    private double[] _coeffs = null;

    //Constructor
    public BackShift(int degree, boolean initial) {
        if (degree < 0) {
            throw new RuntimeException("degree must be non-negative");
        }
        this._degree = degree;
        this._indices = new boolean[_degree + 1];
        for (int j = 0; j <= _degree; ++j) {
            this._indices[j] = initial;
        }
        this._indices[0] = true; // zero index must be true all the time
    }

    public BackShift(boolean[] indices, boolean copyIndices) {
        if (indices == null) {
            throw new RuntimeException("null indices given");
        }
        this._degree = indices.length - 1;
        if (copyIndices) {
            this._indices = new boolean[_degree + 1];
            System.arraycopy(indices, 0, _indices, 0, _degree + 1);
        } else {
            this._indices = indices;
        }
    }

    public int getDegree() {
        return _degree;
    }

    public double[] getCoefficientsFlattened() {
        if (_degree <= 0 || _offsets == null || _coeffs == null) {
            return new double[0];
        }
        int temp = -1;
        for (int offset : _offsets) {
            if (offset > temp) {
                temp = offset;
            }
        }
        final int maxIdx = 1 + temp;
        final double[] flattened = new double[maxIdx];
        for (int j = 0; j < maxIdx; ++j) {
            flattened[j] = 0;
        }
        for (int j = 0; j < _offsets.length; ++j) {
            flattened[_offsets[j]] = _coeffs[j];
        }
        return flattened;
    }

    public void setIndex(int index, boolean enable) {
        _indices[index] = enable;
    }

    public BackShift apply(BackShift another) {
        int mergedDegree = _degree + another._degree;
        boolean[] merged = new boolean[mergedDegree + 1];
        for (int j = 0; j <= mergedDegree; ++j) {
            merged[j] = false;
        }
        for (int j = 0; j <= _degree; ++j) {
            if (_indices[j]) {
                for (int k = 0; k <= another._degree; ++k) {
                    merged[j + k] = merged[j + k] || another._indices[k];
                }
            }
        }
        return new BackShift(merged, false);
    }

    public void initializeParams(boolean includeZero) {
        _indices[0] = includeZero;
        _offsets = null;
        _coeffs = null;
        int nonzeroCount = 0;
        for (int j = 0; j <= _degree; ++j) {
            if (_indices[j]) {
                ++nonzeroCount;
            }
        }
        _offsets = new int[nonzeroCount]; // cannot be 0 as 0-th index is always true
        _coeffs = new double[nonzeroCount];
        int coeffIndex = 0;
        for (int j = 0; j <= _degree; ++j) {
            if (_indices[j]) {
                _offsets[coeffIndex] = j;
                _coeffs[coeffIndex] = 0;
                ++coeffIndex;
            }
        }
    }

    // MAKE SURE to initializeParams before calling below methods
    public int numParams() {
        return _offsets.length;
    }

    public int[] paramOffsets() {
        return _offsets;
    }

    public double getParam(final int paramIndex) {
        int offsetIndex = -1;
        for (int j = 0; j < _offsets.length; ++j) {
            if (_offsets[j] == paramIndex) {
                return _coeffs[j];
            }
        }
        throw new RuntimeException("invalid parameter index: " + paramIndex);
    }

    public double[] getAllParam() {
        return this._coeffs;
    }

    public void setParam(final int paramIndex, final double paramValue) {
        int offsetIndex = -1;
        for (int j = 0; j < _offsets.length; ++j) {
            if (_offsets[j] == paramIndex) {
                offsetIndex = j;
                break;
            }
        }
        if (offsetIndex == -1) {
            throw new RuntimeException("invalid parameter index: " + paramIndex);
        }
        _coeffs[offsetIndex] = paramValue;
    }

    public void copyParamsToArray(double[] dest) {
        System.arraycopy(_coeffs, 0, dest, 0, _coeffs.length);
    }

    public double getLinearCombinationFrom(double[] timeseries, int tsOffset) {
        double linearSum = 0;
        for (int j = 0; j < _offsets.length; ++j) {
            linearSum += timeseries[tsOffset - _offsets[j]] * _coeffs[j];
        }
        return linearSum;
    }
}
