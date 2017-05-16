#!/usr/bin/env bash

if [ "$TRAVIS_BRANCH" = 'master' ] && [ "$TRAVIS_PULL_REQUEST" == 'false' ]; then
    mvn -B release:prepare
    mvn release:rollback
    git tag -d timeseries-forecast-1.0.0
    git push origin :refs/tags/timeseries-forecast-1.0.0
    mvn release:clean
fi

