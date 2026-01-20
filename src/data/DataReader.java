package data;

import tensor.TensorUtil;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader
{
    public static List<Image> getDataSet(String path) {
        List<Image> output = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] lineList = line.split(",");
                double[] imageVector = new double[728];
                for (int i = 0; i < 728; i++)
                    imageVector[i] = Double.parseDouble(lineList[i + 1])/255.0;
                output.add(new Image(Integer.parseInt(lineList[0]), imageVector));
            }

        } catch (Exception e) {

        }
        return output;
    }

    public static List<Image> generateMnistTrainingSet(){
        List<Image> output = new ArrayList<>();
        output.addAll(getDataSet("src/mnist/mnist_train1.csv"));
        output.addAll(getDataSet("src/mnist/mnist_train2.csv"));
        output.addAll(getDataSet("src/mnist/mnist_train3.csv"));
        output.addAll(getDataSet("src/mnist/mnist_train4.csv"));
        output.addAll(getDataSet("src/mnist/mnist_train5.csv"));
        output.addAll(getDataSet("src/mnist/mnist_train6.csv"));
        return output;
    }
}
