package cs446.homework2;

import cs446.weka.classifiers.trees.Id3;
import weka.core.Instances;

import java.util.*;

/**
 * Class DecisionStump
 * Used to train and evaluate Decision Stump
 * Created by Dereck on 0015, February 15, 2017.
 */
public class DecisionStump {
    private List<Id3> stumps = new ArrayList<>();
    private List<List<Integer>> trainData = new ArrayList<>();
    private List<List<Integer>> testData = new ArrayList<>();
    private List<Instances> inst;
    private List<Instances> data;

    /**
     * initStump
     *  initialise DecisionStump with processed data
     */
    public void initStump(String dir) throws Exception{
        DataProcess processor = new DataProcess();
        processor.readData(dir);
        inst = processor.createInstances(dir);
        data = processor.createMerge(inst);
    }

    /**
     * createStump
     *  which create 100 stump according to the given training set and testing #
     */
    public void createStump(Integer test) throws  Exception{
        stumps.clear();
        Instances train = data.get(test);
        int max_number = 4*train.numInstances() / 6;
        //System.out.println(train.numInstances());
        Random rng = new Random();

        for(int i = 0; i < 100; i++){
            Id3 DT = new Id3();
            Set<Integer> generated = new LinkedHashSet<>();
            while(generated.size() < max_number){
                Integer next = rng.nextInt(max_number)+1;
                generated.add(next);
            }
            Instances sample = new Instances(train);        //50% training data set
            for(int j = sample.numInstances()-1; j >= 0; j--){
                if(!generated.contains(j)) sample.delete(j);
            }
//            System.out.println(sample.numInstances());
//            System.out.println(sample.instance(0));
            sample.setClassIndex(sample.numAttributes()-1);
            DT.setMaxDepth(15);
            DT.buildClassifier(sample); //create DicisionStump
            stumps.add(DT);
        }
    }

    /**
     * createData
     *  generate new training data using created DecisionStumps
     */
    public void createData(Integer test) throws  Exception{
        trainData.clear();
        testData.clear();
        Instances train = data.get(test);
        for(int i = 0; i < train.numInstances(); i++){
            List<Integer> line = new ArrayList<>();
            for (Id3 stump : stumps) {
                double predict = stump.classifyInstance(train.instance(i));
                //System.out.println(predict);
                if (predict == 1.0)
                    line.add(1);
                else
                    line.add(-1);
                //line.add((int)predict);
            }
            if((int)train.instance(i).value(260) == 1)
                line.add(1);
            else
                line.add(-1);
            //line.add((int)train.instance(i).value(260));
            trainData.add(line);
        }
        Instances evaluate = inst.get(test);
        for(int i = 0; i < evaluate.numInstances(); i++){
            List<Integer> line = new ArrayList<>();
            for (Id3 stump : stumps) {
                int predict = (int) stump.classifyInstance(evaluate.instance(i));
                //line.add(predict);
                if (predict == 1.0)
                    line.add(1);
                else
                    line.add(-1);
            }
            if((int)evaluate.instance(i).value(260) == 1)
                line.add(1);
            else
                line.add(-1);
            //line.add((int)evaluate.instance(i).value(260));
            testData.add(line);
        }
    }

    /**
     * fiveFoldDecisionStump
     *   perform 5 fold Decision Stump training and testing
     */
    public List<Double> fiveFoldDecitionStump() throws Exception{
        List<Double> ret = new ArrayList<>();
        for(int i = 0; i < 5; i++){
            createStump(i);
            createData(i);
//            System.out.println(trainData.size());
//            System.out.println(testData.size());
            SGD model = new SGD();
//            for(List<Integer> line : trainData){
//                System.out.println(line);
//            }
//            System.out.println("\n\n");
//            for(List<Integer> line : testData){
//                System.out.println(line);
//            }
            model.loadData(trainData, testData);
            //model.train(5);
            ret.add(model.learning());
        }
        return ret;
    }

}
