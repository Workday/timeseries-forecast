/*
 * Copyright (c) 2017-present, Workday, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the LICENSE file in the root repository.
 */

package com.workday.insights.timeseries.arima;

import com.workday.insights.matrix.InsightsMatrix;
import com.workday.insights.matrix.InsightsVector;
import org.junit.Assert;
import org.junit.Test;

public class InsightsMatrixTest {

    @Test
    public void constructorTests() {
        double[][] data = {{3.0, 3.0, 3.0},
            {3.0, 3.0, 3.0},
            {3.0, 3.0, 3.0}};

        InsightsMatrix im1 = new InsightsMatrix(data, false);
        Assert.assertTrue(im1.getNumberOfColumns() == 3);
        Assert.assertTrue(im1.getNumberOfRows() == 3);
        for (int i = 0; i < im1.getNumberOfColumns(); i++) {
            for (int j = 0; j < im1.getNumberOfColumns(); j++) {
                Assert.assertTrue(im1.get(i, j) == 3.0);
            }
        }
        im1.set(0, 0, 0.0);
        Assert.assertTrue(im1.get(0, 0) == 0.0);
        im1.set(0, 0, 3.0);

        InsightsVector iv = new InsightsVector(3, 3.0);
        for (int i = 0; i < im1.getNumberOfColumns(); i++) {
            Assert.assertTrue(im1.timesVector(iv).get(i) == 27.0);
        }
    }

    @Test
    public void solverTestSimple() {
        double[][] A = {
            {2.0}
        };
        double[] B = {4.0};
        double[] solution = {2.0};

        InsightsMatrix im = new InsightsMatrix(A, true);
        InsightsVector iv = new InsightsVector(B, true);

        InsightsVector solved = im.solveSPDIntoVector(iv, -1);
        for (int i = 0; i < solved.size(); i++) {
            Assert.assertTrue(solved.get(i) == solution[i]);
        }
    }

    @Test
    public void solverTestOneSolution() {
        double[][] A = {
            {1.0, 1.0},
            {1.0, 2.0}
        };

        double[] B = {2.0, 16.0};
        double[] solution = {-12.0, 14.0};

        InsightsMatrix im = new InsightsMatrix(A, true);
        InsightsVector iv = new InsightsVector(B, true);

        InsightsVector solved = im.solveSPDIntoVector(iv, -1);
        for (int i = 0; i < solved.size(); i++) {
            Assert.assertTrue(solved.get(i) == solution[i]);
        }
    }

    @Test
    public void timesVectorTestSimple() {
        double[][] A = {
            {1.0, 1.0},
            {2.0, 2.0}
        };

        double[] x = {3.0, 4.0};
        double[] solution = {7.0, 14.0};

        InsightsMatrix im = new InsightsMatrix(A, true);
        InsightsVector iv = new InsightsVector(x, true);

        InsightsVector solved = im.timesVector(iv);
        for (int i = 0; i < solved.size(); i++) {
            Assert.assertTrue(solved.get(i) == solution[i]);

        }
    }

    @Test(expected = RuntimeException.class)
    public void timesVectorTestIncorrectDimension() {
        double[][] A = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0}
        };

        double[] x = {4.0, 4.0, 4.0, 4.0};

        InsightsMatrix im = new InsightsMatrix(A, true);
        InsightsVector iv = new InsightsVector(x, true);

        InsightsVector solved = im.timesVector(iv);
    }

}
