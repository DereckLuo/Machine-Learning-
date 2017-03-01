package cs446.homework2;

import cs446.weka.classifiers.trees.Id3;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.List;

/**
 * Perform 5 fold training and evaluation on the following model
 * 1. SGD
 * 2. Decision Tree full depth
 * 4. Decision Tree 5 levels
 * 5. Decision Tree 10 levels
 * 6. Decision Stumps
 * Created by Dereck on 0015, February 15, 2017.
 */
public class Fivefold {

    public List<Double> fiveFoldSGD(String dir, Double theta){
        List<Double> ret = new ArrayList<>();
        DataProcess processor = new DataProcess();
        processor.readData(dir);
        for (int i = 1; i < 6; i++){
            SGD model = new SGD();
            model.reset(theta);
            List<List<Integer>> trainData = processor.outputTrain(i);
            List<List<Integer>> testData = processor.outputTest(i);
            model.loadData(trainData,testData);
            ret.add(model.learning());
        }
        System.out.println("~~~~ 5 Fold SGD model ~~~~~");
        System.out.println(ret);
        return ret;
    }

    public List<Double> fiveFoldDTfull(String dir) throws Exception{
        List<Double> ret = new ArrayList<>();
        DataProcess processor = new DataProcess();
        List<Instances> inst;
        List<Instances> data;
        Id3 DT = new Id3();

        inst = processor.createInstances(dir);
        System.out.println("Instance size is : " + Integer.toString(inst.size()));

        data = processor.createMerge(inst);

        for(int i = 0; i < 5; i++){
            double sucessor = 0.0;
            Instances train = data.get(i);
            train.setClassIndex(train.numAttributes() - 1);
            //Training the DT
            DT.buildClassifier(train);

            //Evaluating DT
            Instances test = inst.get(i);
            for(int j = 0; j < test.numInstances(); j++) {
                double label = test.instance(j).value(260);
                double predict = DT.classifyInstance(test.instance(j));
                if(label == predict) sucessor ++;
            }
            ret.add(sucessor/test.numInstances());
        }
        System.out.println("~~~~ 5 Fold Full DT model ~~~~~");
        System.out.println(ret);

        return ret;
    }

    public List<Double> fiveFoldDT4(String dir) throws Exception{
        List<Double> ret = new ArrayList<>();
        DataProcess processor = new DataProcess();
        List<Instances> inst;
        List<Instances> data;
        Id3 DT = new Id3();

        inst = processor.createInstances(dir);

        data = processor.createMerge(inst);
        for(int i = 0; i < 5; i++){
            double sucessor = 0.0;
            Instances train = data.get(i);
            train.setClassIndex(train.numAttributes()-1);
            //Training the DT
            DT.setMaxDepth(4);
            DT.buildClassifier(train);

            //Evaluating DT
            Instances test = inst.get(i);
            for(int j = 0; j < test.numInstances(); j++) {
                double label = test.instance(j).value(260);
                double predict = DT.classifyInstance(test.instance(j));
                if(label == predict) sucessor ++;
            }
            //System.out.println(sucessor); System.out.println(test.numInstances());
            ret.add(sucessor/test.numInstances());
        }
        System.out.println("~~~~ 5 Fold 4 level DT model ~~~~~");
        System.out.println(ret);
        return ret;
    }

    public List<Double> fiveFoldDT8(String dir) throws Exception{
        List<Double> ret = new ArrayList<>();
        DataProcess processor = new DataProcess();
        List<Instances> inst;
        List<Instances> data;
        Id3 DT = new Id3();

        inst = processor.createInstances(dir);

        data = processor.createMerge(inst);
        for(int i = 0; i < 5; i++){
            double sucessor = 0.0;
            Instances train = data.get(i);
            train.setClassIndex(train.numAttributes()-1);
            //Training the DT
            DT.setMaxDepth(8);
            DT.buildClassifier(train);
            //Evaluating DT
            Instances test = inst.get(i);
            for(int j = 0; j < test.numInstances(); j++) {
                double label = test.instance(j).value(260);
                double predict = DT.classifyInstance(test.instance(j));
                if(label == predict) sucessor ++;
            }
            //System.out.println(sucessor); System.out.println(test.numInstances());
            ret.add(sucessor/test.numInstances());
        }
        System.out.println("~~~~ 5 Fold 8 level DT model ~~~~~");
        System.out.println(ret);
        return ret;
    }

    public List<Double> fiveFoldStump(String dir) throws Exception{
        DecisionStump stump = new DecisionStump();
        stump.initStump(dir);
        List<Double> ret = stump.fiveFoldDecitionStump();
        System.out.println("~~~~~ 5 Fold Decision Stump model ~~~~~~~");
        System.out.println(ret);
        Double temp = 0.0;
        for(Double num : ret){
            temp += num;
        }
        System.out.println(temp/ret.size());
        return ret;
    }


    /**
     * sgdTune
     *  Given current parameter w, alpha, theta, perform running on parameters alpha and theta
     */
    void sgdTune(String dir){

        double best = 0.0;
        double best_alpha = 0.0; double best_theta = 0.0;

        Integer count = 0;
        for(double alpha = 0.0001; alpha < 0.05; alpha += 0.0005){
            for(double theta = 5; theta < 200; theta += 1){
                List<Double> ret = fiveFoldSGD(dir, theta);
                Double temp = 0.0;
                for(Double num:ret){
                    temp += num;
                }
                Double val = temp/5.0;
                if(val > best){ best = val; best_alpha = alpha; best_theta = theta;}
                count ++;
                if(count % 1000 == 0) System.out.println(count);
            }
        }

        System.out.println(count);
        System.out.println("best alpha is : " + Double.toString(best_alpha));
        System.out.println("best theta is : " + Double.toString(best_theta));
        System.out.println("best result is : " + Double.toString(best));
    }

}
