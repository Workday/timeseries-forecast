# Timeseries-Forecast

This is a Java open source library which provides a time series forecasting functionality. It is an implementation of the [Hannan-Rissanen algorithm](http://www.jstor.org/stable/2241884?seq=1#page_scan_tab_contents "Paper") for additive [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average "Wiki") models. This library is published by the Workday's Syman team, and is used to support basic timeseries forecasting functionalities in some of the Workday products.

[![Build Status](https://travis-ci.org/Workday/timeseries-forecast.svg?branch=master)](https://travis-ci.org/Workday/timeseries-forecast)

How to Use
---

In order to use this library, you need to provide input timeseries data as well as ARIMA parameters. The ARIMA parameters consist of a) non-seasonal parameters `p,d,q` and b) seasonal parameters `P,D,Q,m`. If `D` or `m` is less than 1, then the model is understood to be non-seasonal and the seasonal parameters `P,D,Q,m` will have no effect.

```java
import com.workday.insights.timeseries.arima.Arima;
import com.workday.insights.timeseries.arima.struct.ForecastResult;

// Prepare input timeseries data.
double[] dataArray = new double[] {2, 1, 2, 5, 2, 1, 2, 5, 2, 1, 2, 5, 2, 1, 2, 5};

// Set ARIMA model parameters.
int p = 3;
int d = 0;
int q = 3;
int P = 1;
int D = 1;
int Q = 0;
int m = 0;
int forecastSize = 1;

// Obtain forecast result. The structure contains forecasted values and performance metric etc.
ForecastResult forecastResult = Arima.forecast_arima(dataArray, forecastSize, p, d, q, P, D, Q, m);

// Read forecast values
double[] forecastData = forecastResult.getForecast(); // in this example, it will return { 2 }

// You can obtain upper- and lower-bounds of confidence intervals on forecast values.
// By default, it computes at 95%-confidence level. This value can be adjusted in ForecastUtil.java
double[] uppers = forecastResult.getForecastUpperConf();
double[] lowers = forecastResult.getForecastLowerConf();

// You can also obtain the root mean-square error as validation metric.
double rmse = forecastResult.getRMSE();

// It also provides the maximum normalized variance of the forecast values and their confidence interval.
double maxNormalizedVariance = forecastResult.getMaxNormalizedVariance();

// Finally you can read log messages.
String log = forecastResult.getLog();
```

How to Build
---
This library uses Maven as its build tool.

```java
// Compile the source code of the project.
mvn compile

// To generate javadocs
mvn javadoc:javadoc

// To generate a site for the current project
mvn site

// Take the compiled code and package it
mvn package

// Install the package into the local repository, which can be used as a dependency in other projects locally.
mvn install
```

Dependencies
---

The library has the following dependencies:
```
JUnit 4.12
```

Authors
---

Here is the [Contributors List](CONTRIBUTORS.md) for the timeseries-forecast library.
Please note that the project was developed and ported from an internal repository. Therefore, the commit record does not reflect the full history of the project.

License
---

Copyright 2017 Workday, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
