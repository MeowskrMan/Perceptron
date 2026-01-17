import data.DataReader;
import data.Image;
import layer.PerceptronLayer;
import tensor.TensorUtil;

import java.util.List;

void main() {
    List<Image> dataSet = DataReader.getDataSet("src/mnist/mnist_test.csv");
    PerceptronLayer layer = new PerceptronLayer(10, 728, 420);
    Image image;
    double[] expected = new double[10];
    double[] input;
    double[] output;
    double[] foo;
    for(int i = 0; i < 200; i++){
        image = dataSet.get(i);
        input = image.getImageVector();
        output = layer.forwardPass(input);
        for(int j = 0; j < 10; j++){
            if(image.getLabel()-1 == j){
                expected[j] = 1;
            }
            else {
                expected[j] = 0;
            }
        }
        layer.backwardPass(TensorUtil.derivativeLoss(expected, output));
        System.out.println(TensorUtil.loss(expected, output));
    }
}