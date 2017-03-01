package cs446.homework2;


import java.util.List;

/**
 * Created by Dereck on 0013, February 13, 2017.
 * CS 446 Homework 2
 * Decision Tree Training Main class
 */
public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("CS 446 Homework 2 Starts !");
        String work_dir = "E:\\UIUC\\Spring2017\\CS446 Machine Learning\\Homework 2\\data\\badges";

        generateARFF(work_dir);

        Fivefold eval = new Fivefold();
        StatAnalysis stat = new StatAnalysis();
        double temp = 0;

        /* SGD 5 fold cross validation */
        Double theta = 0.001;
        List<Double> SGDresult = eval.fiveFoldSGD(work_dir, theta);
        //eval.sgdTune(work_dir);
        for(Double val : SGDresult){
            temp += val;
        }
        System.out.println(temp/SGDresult.size());
        stat.tStat(SGDresult);

        /* full DT 5 fold cross validation */
        List<Double> fullDT = eval.fiveFoldDTfull(work_dir);
        temp = 0;
        for(Double val : fullDT){
            temp += val;
        }
        System.out.println(temp/fullDT.size());
        stat.tStat(fullDT);

        /* 4 level DT 5 fold corss validation */
        List<Double> DT4 = eval.fiveFoldDT4(work_dir);
        temp = 0;
        for(Double val : DT4){
            temp += val;
        }
        System.out.println(temp/DT4.size());
        stat.tStat(DT4);

        /* 8 level DT 5 fold cross validation */
        List<Double> DT8 = eval.fiveFoldDT8(work_dir);
        temp = 0;
        for(Double val : DT8){
            temp += val;
        }
        System.out.println(temp/DT8.size());
        stat.tStat(DT8);

        List<Double> DS = eval.fiveFoldStump(work_dir);
        stat.tStat(DS);



    }

    /**
     * generateARFF :
     *  Function to generate all ARFF files from input fold file
     *  using the given function
     */
    private static void generateARFF(String dir) throws Exception{

        String read_file = dir + "\\badges.modified.data.fold";
        String write_file = dir + "\\bages";

        FeatureGenerator featureGenerator = new FeatureGenerator();

        for (int i = 1; i <= 5; i++){
            String read_fileName = read_file + Integer.toString(i);
            String write_fileName = write_file + Integer.toString(i) + ".arff";
            String[] arguments = new String[]{read_fileName, write_fileName};
            FeatureGenerator.main(arguments);
        }
    }
}
