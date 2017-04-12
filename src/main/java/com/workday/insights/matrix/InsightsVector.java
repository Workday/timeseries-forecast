/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.matrix;

import java.io.Serializable;

/**
 * InsightsVector
 *
 * <p> Vector of double entries </p>
 */
public class InsightsVector implements Serializable {

    private static final long serialVersionUID = 43L;

    protected int _m = -1;
    protected double[] _data = null;
    protected boolean _valid = false;

    //=====================================================================
    // Constructors
    //=====================================================================

    /**
     * Constructor for InsightVector
     *
     * @param m size of the vector
     * @param value initial value for all entries
     */
    public InsightsVector(int m, double value) {
        if (m <= 0) {
            throw new RuntimeException("[InsightsVector] invalid size");
        } else {
            _data = new double[m];
            for (int j = 0; j < m; ++j) {
                _data[j] = value;
            }
            _m = m;
            _valid = true;
        }
    }

    /**
     * Constructor for InsightVector
     *
     * @param data 1-dimensional double array with pre-populated values
     * @param deepCopy if TRUE, allocated new memory space and copy data over
     *                 if FALSE, re-use the given memory space and overwrites on it
     */
    public InsightsVector(double[] data, boolean deepCopy) {
        if (data == null || data.length == 0) {
            throw new RuntimeException("[InsightsVector] invalid data");
        } else {
            _m = data.length;
            if (deepCopy) {
                _data = new double[_m];
                System.arraycopy(data, 0, _data, 0, _m);
            } else {
                _data = data;
            }
            _valid = true;
        }
    }
    //=====================================================================
    // END of Constructors
    //=====================================================================
    //=====================================================================
    // Helper Methods
    //=====================================================================

    /**
     * Create and allocate memory for a new copy of double array of current elements in the vector
     *
     * @return the new copy
     */
    public double[] deepCopy() {
        double[] dataDeepCopy = new double[_m];
        System.arraycopy(_data, 0, dataDeepCopy, 0, _m);
        return dataDeepCopy;
    }
    //=====================================================================
    // END of Helper Methods
    //=====================================================================
    //=====================================================================
    // Getters & Setters
    //=====================================================================

    /**
     * Getter for the i-th element in the vector
     *
     * @param i element index
     * @return the i-th element
     */
    public double get(int i) {
        if (!_valid) {
            throw new RuntimeException("[InsightsVector] invalid Vector");
        } else if (i >= _m) {
            throw new IndexOutOfBoundsException(
                String.format("[InsightsVector] Index: %d, Size: %d", i, _m));
        }
        return _data[i];
    }

    /**
     * Getter for the size of the vector
     *
     * @return size of the vector
     */
    public int size() {
        if (!_valid) {
            throw new RuntimeException("[InsightsVector] invalid Vector");
        }

        return _m;
    }

    /**
     * Setter to modify a element in the vector
     *
     * @param i element index
     * @param val new value
     */
    public void set(int i, double val) {
        if (!_valid) {
            throw new RuntimeException("[InsightsVector] invalid Vector");
        } else if (i >= _m) {
            throw new IndexOutOfBoundsException(
                String.format("[InsightsVector] Index: %d, Size: %d", i, _m));
        }
        _data[i] = val;
    }

    //=====================================================================
    // END of Getters & Setters
    //=====================================================================
    //=====================================================================
    // Basic Linear Algebra operations
    //=====================================================================

    /**
     * Perform dot product operation with another vector of the same size
     *
     * @param vector vector of the same size
     * @return dot product of the two vector
     */
    public double dot(InsightsVector vector) {
        if (!_valid || !vector._valid) {
            throw new RuntimeException("[InsightsVector] invalid Vector");
        } else if (_m != vector.size()) {
            throw new RuntimeException("[InsightsVector][dot] invalid vector size.");
        }

        double sumOfProducts = 0;
        for (int i = 0; i < _m; i++) {
            sumOfProducts += _data[i] * vector.get(i);
        }
        return sumOfProducts;
    }
    //=====================================================================
    // END of Basic Linear Algebra operations
    //=====================================================================
}
