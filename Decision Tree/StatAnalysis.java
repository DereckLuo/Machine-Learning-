package cs446.homework2;

import java.util.ArrayList;
import java.util.List;

/**
 * Class StatAnalysis:
 *  Statistically analyze the given data
 * Created by Dereck on 0016, February 16, 2017.
 */
public class StatAnalysis {
    /**
     * tStat:
     *  Given a list of input number, analyze with t statistic
     *  compute and output 99% confidence interval
     */
    public void tStat(List<Double> input){
        Double sum = 0.0;
        for (Double num : input){
            sum += num;
        }
        Double mean = sum/input.size();
        Double df = input.size() - 1.0;
        List<Double> rms = new ArrayList<>();
        for( Double num : input){
            rms.add( (num - mean) * (num - mean));
        }
        Double var = 0.0;
        for(Double num : rms){
            var += num;
        }
        var = var / rms.size();
        Double std = Math.sqrt(var);
        Double t_value = 4.604;

        Double lower_bound = -t_value * (std/Math.sqrt(input.size())) + mean;
        Double upper_bound = t_value * (std/Math.sqrt(input.size())) + mean;
        System.out.println("99% confidence integrval is at : [" + Double.toString(lower_bound)
         + " , " + Double.toString(upper_bound) + "]");
    }


}
