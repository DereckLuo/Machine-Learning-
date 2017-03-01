package cs446.homework2;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class DataProcess
 * Given the direct path of all files
 * Given the portion of test data
 * output training data set and testing data set
 * Created by Dereck on 0014, February 14, 2017.
 */
class DataProcess {
    private List<List<List<Integer>>> data_array = new ArrayList<>();
    /**
     * readData
     * which filling up data_array according to the 5 input files
     */
    void readData(String dir){
        for (int i = 1; i < 6; i++){    //read in 5 data files
            boolean data = false;

            List<List<Integer>> book = new ArrayList<>();
            try(BufferedReader br = new BufferedReader(new FileReader(dir + "\\bages" + Integer.toString(i) + ".arff"))) {
                for(String line; (line = br.readLine()) != null; ) {
                    if (data){  //case reading the line with data
                        List<Integer> bline = new ArrayList<>();
                        line = line.replace(",","");
                        for(int j = 0; j < line.length()-1; j++){
                            bline.add((int) line.charAt(j) -48);
                        }
                        if(line.charAt(line.length()-1) == '+')
                            bline.add(1);
                        else
                            bline.add(-1);
                        book.add(bline);
                    }
                    if (line.equals("@data")) data = true;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            //System.out.println(book.size());
            data_array.add(book);
        }
    }

    /**
     * outputTrain
     *  function output the training data set according to which portion is used for testing
     */
    List<List<Integer>> outputTrain(Integer test){
        List<List<Integer>> ret = new ArrayList<>();
        for(int i = 0; i < 5; i++){
            if((i+1) != test){
                for(int j = 0; j < data_array.get(i).size(); j++){
                    List<Integer> line = new ArrayList<>();
                    for(int k = 0; k < data_array.get(i).get(j).size(); k++){
                        line.add(data_array.get(i).get(j).get(k));
                    }
                    ret.add(line);
                }
            }
        }
        return ret;
    }

    /**
     * outputTest
     *  function output the testing data set according to which portion is used for testing
     */
    List<List<Integer>> outputTest(Integer test){
        List<List<Integer>> ret = new ArrayList<>();
        Integer idx = test-1;
        for(int i = 0; i < data_array.get(idx).size(); i++){
            List<Integer> line = new ArrayList<>();
            for(int j = 0; j < data_array.get(idx).get(i).size(); j++){
                line.add(data_array.get(idx).get(i).get(j));
            }
            ret.add(line);
        }
        return ret;
    }

    /**
     * merge instances:
     *  Code given from Piazza
     */
    public static Instances merge(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }

    /**
     * createInstances
     *  Function to create 5 instances given the path of .arff files
     */
    public List<Instances> createInstances(String dir) throws IOException {
        List<Instances> inst = new ArrayList<>();
        for(int i = 1; i < 6; i++){
            BufferedReader reader = new BufferedReader(
                    new FileReader(dir + "\\bages" + Integer.toString(i) + ".arff"));
            Instances temp = new Instances(reader);
            inst.add(temp);
        }
        return inst;
    }

    /**
     * createMerge
     *  Function to create merged training data given the path of .arff files and
     */
    public List<Instances> createMerge(List<Instances> inst) throws Exception {
        List<Instances> data = new ArrayList<>();

        //Create merge instances and insert into data array
        Instances temp1 = merge(inst.get(1), inst.get(2));    //testing 1
        Instances temp2 = merge(inst.get(3), inst.get(4));
        data.add(merge(temp1,temp2));

        temp1 = merge(inst.get(0),inst.get(2));               //testing 2
        data.add(merge(temp1,temp2));

        temp1 = merge(inst.get(0),inst.get(1));               //testing 3
        data.add(merge(temp1, temp2));

        temp2 = merge(inst.get(2),inst.get(4));               //testing 4
        data.add(merge(temp1,temp2));

        temp2 = merge(inst.get(2),inst.get(3));               //testing 5
        data.add(merge(temp1,temp2));

        return data;
    }
}
