#!/usr/bin/env bash

DATA_PATH=data/downstream
STS_DATA_PATH=$DATA_PATH/STS

mkdir -p $STS_DATA_PATH

# Download STS2012
URL='http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip'
FILENAME='STS2012-en-test.zip'
FOLDER_NAME='STS12-en-test'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    unzip $STS_DATA_PATH/$FILENAME -d $FOLDER_PATH && \
    mv $FOLDER_PATH/*/*.txt $FOLDER_PATH && \
    rm -rf $FOLDER_PATH/*/ && \
    rm -rf $STS_DATA_PATH/$FILENAME

# Download STS2013
URL='http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip'
FILENAME='STS2013-en-test.zip'
FOLDER_NAME='STS13-en-test'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    unzip $STS_DATA_PATH/$FILENAME -d $FOLDER_PATH && \
    mv $FOLDER_PATH/*/*.txt $FOLDER_PATH && \
    rm -rf $FOLDER_PATH/*/ && \
    rm -rf $STS_DATA_PATH/$FILENAME

# Download STS2014
URL='http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip'
FILENAME='STS2014-en-test.zip'
FOLDER_NAME='STS14-en-test'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    unzip $STS_DATA_PATH/$FILENAME -d $FOLDER_PATH && \
    mv $FOLDER_PATH/*/*.txt $FOLDER_PATH && \
    rm -rf $FOLDER_PATH/*/ && \
    rm -rf $STS_DATA_PATH/$FILENAME

# Download STS2015
URL='http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip'
FILENAME='STS2015-en-test.zip'
FOLDER_NAME='STS15-en-test'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    unzip $STS_DATA_PATH/$FILENAME -d $FOLDER_PATH && \
    mv $FOLDER_PATH/*/*.txt $FOLDER_PATH && \
    rm -rf $FOLDER_PATH/*/ && \
    rm -rf $STS_DATA_PATH/$FILENAME

# Download STS2016
URL='http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip'
FILENAME='STS2016-en-test.zip'
FOLDER_NAME='STS16-en-test'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    unzip $STS_DATA_PATH/$FILENAME -d $FOLDER_PATH && \
    mv $FOLDER_PATH/*/*.txt $FOLDER_PATH && \
    rm -rf $FOLDER_PATH/*/ && \
    rm -rf $STS_DATA_PATH/$FILENAME

# Download STS-Benchmark
URL='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
FILENAME='Stsbenchmark.tar.gz'
FOLDER_NAME='STSBenchmark'
FOLDER_PATH=$STS_DATA_PATH/$FOLDER_NAME
wget $URL -P $STS_DATA_PATH && \
    tar -xvf $STS_DATA_PATH/$FILENAME -C $STS_DATA_PATH && \
    mv $STS_DATA_PATH/stsbenchmark $STS_DATA_PATH/STSBenchmark && \
    rm -rf $STS_DATA_PATH/$FILENAME
