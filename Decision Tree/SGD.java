package cs446.homework2;

import java.util.*;

/**
 * Created by Dereck on 0014, February 14, 2017.
 * Stochastic Gradient Descent:
 * Implementing using LMS loss funciton
 */
class SGD {

    private List<List<Integer>> trainData = new ArrayList<>();
    private List<List<Integer>> testData = new ArrayList<>();

    private List<Double> w = new ArrayList<>();    //parameter

    private Double theta = 0.001;


    /**
     * loadData
     * Given input training data and testing data, load into private variable
     * Generate initial parameter w
     */
    void loadData(List<List<Integer>> train, List<List<Integer>> test) {
        for (List<Integer> aTrain : train) {
            List<Integer> line = new ArrayList<>();
            for (Integer anATrain : aTrain) {
                line.add(anATrain);
            }
            trainData.add(line);
        }

        for (int i = 0; i < test.size(); i++) {
            List<Integer> line = new ArrayList<>();
            for (int j = 0; j < train.get(i).size(); j++) {
                line.add(test.get(i).get(j));
            }
            testData.add(line);
        }
        for (int i = 0; i < train.get(0).size()-1; i++) {
            w.add(0.0);
        }
    }

    void reset(Double t){
        for(int i = 0; i < w.size(); i++){
            w.set(i,0.0);
        }
        theta = t;
    }

    void printData() {
        System.out.println("~~~~~~Training Data Set~~~~~~~~");
        System.out.println("=========== Training Data Size : " + Integer.toString(trainData.size()));
        for (List<Integer> aTrainData : trainData) {
            for (Integer anATrainData : aTrainData) {
                System.out.print(Integer.toString(anATrainData) + " ");
            }
            System.out.println("");
        }
        System.out.println("~~~~~~Testing Data Set~~~~~~~~");
        System.out.println("============ Testing Data Size : " + Integer.toString((testData.size())));
        for (List<Integer> aTestData : testData) {
            for (Integer anATestData : aTestData) {
                System.out.print(Integer.toString(anATestData) + " ");
            }
            System.out.println("");
        }
        System.out.println("~~~~~~Parameter~~~~~~~~~");
        System.out.println(w);

    }

    /**
     * Training LMS model on training data set
     */
    void train(Double c ){
        Integer max_number = 50; //number of steps
        Random rng = new Random();
        Set<Integer> generated = new LinkedHashSet<>();
        while(generated.size() < max_number){
            Integer next = rng.nextInt(max_number)+1;
            generated.add(next);
        }
        for(Integer num : generated){
            double temp = 0;
            List<Integer> x = trainData.get(num);
            Integer y = x.get(w.size());
            for(int i = 0; i < w.size(); i++){  //w_t * x_t
                temp +=  (w.get(i) * (double)x.get(i));
            }
            temp = (double)y - temp; // y_t - (w_t*x_t)
            for(int i = 0; i < w.size(); i++){
                double new_w = w.get(i) + (1.0/(1000+c))*temp*(double)x.get(i);
                w.set(i, new_w);
            }
        }
    }

    /**
     * Learning:
     *  Perform 100 step per season, each season do 1 evaluation
     *  When reaching threshold theta, stop learning
     */
    double learning(){
        double prev = 100.0; double post = 0.0;
        Double count = 0.0;
        //while(count < 100){
            count ++;
        while(Math.abs(prev-post) > theta){
            train(count);
            prev = post; post = test();
        }
        double result = test();
        return result;
    }

    /**
     * Testing LMS model on testing data set
     * Output and printout correction % score
     */
    double test(){
        Integer success = 0; Integer fail = 0;
        for (List<Integer> aTestData : testData) {
            double temp = 0;
            Integer label = aTestData.get(w.size());
            for (int j = 0; j < w.size(); j++) {  //w * x
                temp += (w.get(j) * (double) aTestData.get(j));
            }

            if (temp > 0 && label == 1) success++;
            if (temp > 0 && label == -1) fail ++;
            if (temp <= 0 && label == -1) success++;
            if (temp <= 0 && label == 1) fail++;

        }
        return (double)success/(double)testData.size();
    }





}
