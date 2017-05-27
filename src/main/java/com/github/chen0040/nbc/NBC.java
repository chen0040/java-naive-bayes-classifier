package com.github.chen0040.nbc;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.CountRepository;
import com.github.chen0040.data.utils.discretizers.KMeansDiscretizer;

import java.util.*;
import java.util.logging.Logger;


/**
 * Created by memeanalytics on 18/8/15.
 */
public class NBC {
    private final static Logger logger = Logger.getLogger(String.valueOf(NBC.class));
    private CountRepository model = new CountRepository();
    private KMeansDiscretizer inputDiscretizer = new KMeansDiscretizer();
    private final List<String> classLabels = new ArrayList<>();
    

    protected boolean isValidTrainingSample(DataRow tuple){
        return !tuple.getCategoricalTargetColumnNames().isEmpty();
    }

    public void copy(NBC that){

        model = that.model == null ? null : that.model.makeCopy();
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        NBC clone = (NBC)super.clone();
        clone.copy(this);

        return clone;
    }

    public NBC(){

    }

    public CountRepository getModel(){
        return model;
    }

    public String classify(DataRow tuple) {
        Map<String, Double> scores = getScores(tuple);

        double maxScore = 0;
        String bestLabel = null;
        for(String classLabel : scores.keySet()){
            double score = scores.get(classLabel);
            if(maxScore < score){
                maxScore = score;
                bestLabel = classLabel;
            }
        }

        return bestLabel;

    }

    public HashMap<String, Double> getScores(DataRow tuple)
    {
        HashMap<String, Double> scores = new HashMap<>();
        for(String classLabel : classLabels) {
            String classEventName = "ClassLabel="+classLabel;

            double score = 1;

            double pC = model.getProbability(classEventName);
            score *= pC;

            // categorical columns
            List<String> columns = tuple.getCategoricalColumnNames();
            int n = columns.size();
            for (int i = 0; i < n; ++i) {
                String variableName = columns.get(i);
                String value = tuple.getCategoricalCell(variableName);
                String eventName = variableName + "=" + value;

                double pXiC = model.getConditionalProbability(classEventName, eventName);
                score *= pXiC;
            }

            // numerical columns
            columns = tuple.getColumnNames();
            n = columns.size();
            for (int i = 0; i < n; ++i) {
                String variableName = columns.get(i);
                double value = tuple.getCell(variableName);
                int label = inputDiscretizer.discretize(value, variableName);
                String eventName = variableName + "=" + label;

                double pXiC = model.getConditionalProbability(classEventName, eventName);
                score *= pXiC;
            }

            scores.put(classLabel, score);
        }

        return scores;
    }



    private void initializeClassLabels(DataFrame batch){
        Set<String> set = new HashSet<>();
        int m = batch.rowCount();

        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            if(isValidTrainingSample(tuple)) {
                set.add(tuple.categoricalTarget());
            }
        }

        classLabels.clear();
        classLabels.addAll(set);
    }

    public void fit(DataFrame batch) {
        

        batch = batch.filter(this::isValidTrainingSample);

        inputDiscretizer.fit(batch);

        model = new CountRepository();

        initializeClassLabels(batch);

        int m = batch.rowCount();

        for(int i=0; i < m ; ++i){
            DataRow tuple = batch.row(i);

            String classEventName = "ClassLabel=" + tuple.categoricalTarget();

            // categorical columns
            List<String> columnNames = tuple.getCategoricalColumnNames();
            int n = columnNames.size();
            for (int j = 0; j < n; ++j) {
                String variableName = columnNames.get(i);
                String value = tuple.getCategoricalCell(variableName);
                String eventName = variableName + "=" + value;
                model.addSupportCount(classEventName, eventName);
                model.addSupportCount(classEventName);
                model.addSupportCount();
            }

            // numerical columns
            columnNames = tuple.getColumnNames();
            n = columnNames.size();
            for (int j = 0; j < n; ++j) {
                String variableName = columnNames.get(i);
                double value = tuple.getCell(variableName);
                int label = inputDiscretizer.discretize(value, variableName);
                String eventName = variableName + "=" + label;
                model.addSupportCount(classEventName, eventName);
                model.addSupportCount(classEventName);
                model.addSupportCount();
            }
        }
    }

}
