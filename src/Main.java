import data.DataReader;
import data.Image;
import layer.OutputLayer;
import network.NeuralNetwork;
import tensor.TensorUtil;


void main() {
    NeuralNetwork net = new NeuralNetwork(728,10,1, 16, 112312234, 0.03);
    List<Image> trainingSet = DataReader.generateMnistTrainingSet();
    net.train(trainingSet);

}