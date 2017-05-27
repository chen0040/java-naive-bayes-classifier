# java-naive-bayes-classifier
Package provides java implementation of naive bayes classifier

[![Build Status](https://travis-ci.org/chen0040/java-naive-bayes-classifier.svg?branch=master)](https://travis-ci.org/chen0040/java-naive-bayes-classifier) [![Coverage Status](https://coveralls.io/repos/github/chen0040/java-naive-bayes-classifier/badge.svg?branch=master)](https://coveralls.io/github/chen0040/java-naive-bayes-classifier?branch=master) 


# Features

* Handle both numerical and categorical inputs

# Install

Add the following dependency to your POM file

# Usage

To train the NBC:

```java
nbc.fit(trainingData);
```

To use NBC for classification:

```java
String predicted = nbc.classify(dataRow);
```

The trainingData object is an instance of data frame consisting of data rows (Please refers to this [link](https://github.com/chen0040/java-data-frame) to find out how to store data into a data frame)

The sample code below shows how to use NBC to solves the classification problem "heart_scale".

```java
InputStream inputStream = new FileInputStream("heart_scale");

DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();


dataFrame.unlock();
for(int i=0; i < dataFrame.rowCount(); ++i){
 DataRow row = dataFrame.row(i);
 row.setCategoricalTargetCell("category-label", "" + row.target());
}
dataFrame.lock();

NBC svc = new NBC();
svc.fit(dataFrame);

for(int i = 0; i < dataFrame.rowCount(); ++i){
 DataRow row = dataFrame.row(i);
 String predicted_label = svc.classify(row);
 System.out.println("predicted: "+predicted_label+"\texpected: "+row.categoricalTarget());
}
```


