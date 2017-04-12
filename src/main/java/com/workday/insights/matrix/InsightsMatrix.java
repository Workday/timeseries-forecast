/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.matrix;

import java.io.Serializable;

/**
 * InsightsMatrix
 *
 * <p>
 * A small set of linear algebra methods to be used in ARIMA
 * </p>
 */
public class InsightsMatrix implements Serializable {

    public static final long serialVersionUID = 42L;

    // Primary
    protected int _m = -1;
    protected int _n = -1;
    protected double[][] _data = null;
    protected boolean _valid = false;

    // Secondary
    protected boolean _cholZero = false;
    protected boolean _cholPos = false;
    protected boolean _cholNeg = false;
    protected double[] _cholD = null;
    protected double[][] _cholL = null;

    //=====================================================================
    // Constructors
    //=====================================================================

    /**
     * Constructor for InsightsMatrix
     *
     * @param data 2-dimensional double array with pre-populated values
     * @param makeDeepCopy if TRUE, allocated new memory space and copy data over
     *                     if FALSE, re-use the given memory space and overwrites on it
     */
    public InsightsMatrix(double[][] data, boolean makeDeepCopy) {
        if (_valid = isValid2D(data)) {
            _m = data.length;
            _n = data[0].length;
            if (!makeDeepCopy) {
                _data = data;
            } else {
                _data = copy2DArray(data);
            }

        }
    }

    //=====================================================================
    // END of Constructors
    //=====================================================================
    //=====================================================================
    // Helper Methods
    //=====================================================================

    /**
     * Determine whether a 2-dimensional array is in valid matrix format.
     *
     * @param matrix 2-dimensional double array
     * @return TRUE, matrix is in valid format
     *         FALSE, matrix is not in valid format
     */
    private static boolean isValid2D(double[][] matrix) {
        boolean result = true;
        if (matrix == null || matrix[0] == null || matrix[0].length == 0) {
            throw new RuntimeException("[InsightsMatrix][constructor] null data given");
        } else {
            int row = matrix.length;
            int col = matrix[0].length;
            for (int i = 1; i < row; ++i) {
                if (matrix[i] == null || matrix[i].length != col) {
                    result = false;
                }
            }
        }

        return result;
    }

    /**
     * Create a copy of 2-dimensional double array by allocating new memory space and copy data over
     *
     * @param source source 2-dimensional double array
     * @return new copy of the source 2-dimensional double array
     */
    private static double[][] copy2DArray(double[][] source) {
        if (source == null) {
            return null;
        } else if (source.length == 0) {
            return new double[0][];
        }

        int row = source.length;
        double[][] target = new double[row][];
        for (int i = 0; i < row; i++) {
            if (source[i] == null) {
                target[i] = null;
            } else {
                int rowLength = source[i].length;
                target[i] = new double[rowLength];
                System.arraycopy(source[i], 0, target[i], 0, rowLength);
            }
        }
        return target;
    }

    //=====================================================================
    // END of Helper Methods
    //=====================================================================
    //=====================================================================
    // Getters & Setters
    //=====================================================================

    /**
     * Getter for number of rows of the matrix
     *
     * @return number of rows
     */
    public int getNumberOfRows() {
        return _m;
    }

    /**
     * Getter for number of columns of the matrix
     *
     * @return number of columns
     */
    public int getNumberOfColumns() {
        return _n;
    }

    /**
     * Getter for a particular element in the matrix
     *
     * @param i i-th row
     * @param j j-th column
     * @return the element from the i-th row and j-th column from the matrix
     */
    public double get(int i, int j) {
        return _data[i][j];
    }

    /**
     * Setter to modify a particular element in the matrix
     *
     * @param i i-th row
     * @param j j-th column
     * @param val new value
     */
    public void set(int i, int j, double val) {
        _data[i][j] = val;
    }

    //=====================================================================
    // END of Getters & Setters
    //=====================================================================
    //=====================================================================
    // Basic Linear Algebra operations
    //=====================================================================

    /**
     * Multiply a InsightMatrix (n x m) by a InsightVector (m x 1)
     *
     * @param v a InsightVector
     * @return a InsightVector of dimension (n x 1)
     */
    public InsightsVector timesVector(InsightsVector v) {
        if (!_valid || !v._valid || _n != v._m) {
            throw new RuntimeException("[InsightsMatrix][timesVector] size mismatch");
        }
        double[] data = new double[_m];
        double dotProduc;
        for (int i = 0; i < _m; ++i) {
            InsightsVector rowVector = new InsightsVector(_data[i], false);
            dotProduc = rowVector.dot(v);
            data[i] = dotProduc;
        }
        return new InsightsVector(data, false);
    }

    // More linear algebra operations

    /**
     * Compute the Cholesky Decomposition
     *
     * @param maxConditionNumber maximum condition number
     * @return TRUE, if the process succeed
     *         FALSE, otherwise
     */
    private boolean computeCholeskyDecomposition(final double maxConditionNumber) {
        _cholD = new double[_m];
        _cholL = new double[_m][_n];
        int i;
        int j;
        int k;
        double val;
        double currentMax = -1;
        // Backward marching method
        for (j = 0; j < _n; ++j) {
            val = 0;
            for (k = 0; k < j; ++k) {
                val += _cholD[k] * _cholL[j][k] * _cholL[j][k];
            }
            double diagTemp = _data[j][j] - val;
            final int diagSign = (int) (Math.signum(diagTemp));
            switch (diagSign) {
                case 0:    // singular diagonal value detected
                    if (maxConditionNumber < -0.5) { // no bound on maximum condition number
                        _cholZero = true;
                        _cholL = null;
                        _cholD = null;
                        return false;
                    } else {
                        _cholPos = true;
                    }
                    break;
                case 1:
                    _cholPos = true;
                    break;
                case -1:
                    _cholNeg = true;
                    break;
            }
            if (maxConditionNumber > -0.5) {
                if (currentMax <= 0.0) { // this is the first time
                    if (diagSign == 0) {
                        diagTemp = 1.0;
                    }
                } else { // there was precedent
                    if (diagSign == 0) {
                        diagTemp = Math.abs(currentMax / maxConditionNumber);
                    } else {
                        if (Math.abs(diagTemp * maxConditionNumber) < currentMax) {
                            diagTemp = diagSign * Math.abs(currentMax / maxConditionNumber);
                        }
                    }
                }
            }
            _cholD[j] = diagTemp;
            if (Math.abs(diagTemp) > currentMax) {
                currentMax = Math.abs(diagTemp);
            }
            _cholL[j][j] = 1;
            for (i = j + 1; i < _m; ++i) {
                val = 0;
                for (k = 0; k < j; ++k) {
                    val += _cholD[k] * _cholL[j][k] * _cholL[i][k];
                }
                val = ((_data[i][j] + _data[j][i]) / 2 - val) / _cholD[j];
                _cholL[j][i] = val;
                _cholL[i][j] = val;
            }
        }
        return true;
    }

    /**
     * Solve SPD(Symmetric positive definite) into vector
     *
     * @param b vector
     * @param maxConditionNumber maximum condition number
     * @return solution vector of SPD
     */
    public InsightsVector solveSPDIntoVector(InsightsVector b, final double maxConditionNumber) {
        if (!_valid || b == null || _n != b._m) {
            // invalid linear system
            throw new RuntimeException(
                "[InsightsMatrix][solveSPDIntoVector] invalid linear system");
        }
        if (_cholL == null) {
            // computing Cholesky Decomposition
            this.computeCholeskyDecomposition(maxConditionNumber);
        }
        if (_cholZero) {
            // singular matrix. returning null
            return null;
        }

        double[] y = new double[_m];
        double[] bt = new double[_n];
        int i;
        int j;
        for (i = 0; i < _m; ++i) {
            bt[i] = b._data[i];
        }
        double val;
        for (i = 0; i < _m; ++i) {
            val = 0;
            for (j = 0; j < i; ++j) {
                val += _cholL[i][j] * y[j];
            }
            y[i] = bt[i] - val;
        }
        for (i = _m - 1; i >= 0; --i) {
            val = 0;
            for (j = i + 1; j < _n; ++j) {
                val += _cholL[i][j] * bt[j];
            }
            bt[i] = y[i] / _cholD[i] - val;
        }
        return new InsightsVector(bt, false);
    }

    /**
     * Computu the product of the matrix (m x n) and its transpose (n x m)
     *
     * @return matrix of size (m x m)
     */
    public InsightsMatrix computeAAT() {
        if (!_valid) {
            throw new RuntimeException("[InsightsMatrix][computeAAT] invalid matrix");
        }
        final double[][] data = new double[_m][_m];
        for (int i = 0; i < _m; ++i) {
            final double[] rowI = _data[i];
            for (int j = 0; j < _m; ++j) {
                final double[] rowJ = _data[j];
                double temp = 0;
                for (int k = 0; k < _n; ++k) {
                    temp += rowI[k] * rowJ[k];
                }
                data[i][j] = temp;
            }
        }
        return new InsightsMatrix(data, false);
    }
    //=====================================================================
    // END of Basic Linear Algebra operations
    //=====================================================================
}
