package network;

import data.DataReader;
import data.Image;
import layer.HiddenLayer;
import layer.Layer;
import layer.OutputLayer;
import tensor.TensorUtil;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    List<Layer> network = new ArrayList<>();

    public NeuralNetwork(int numberOfInputs, int numberOfOutputs, int numberOfHiddenLayers, int neuronCount, long SEED, double learningRate){
        if(numberOfHiddenLayers == 0){
            network.add(new OutputLayer(numberOfOutputs, numberOfInputs, SEED, learningRate));
        }

        else {
            network.add(new HiddenLayer(neuronCount, numberOfInputs, SEED, learningRate));
            for (int i = 1; i < numberOfHiddenLayers; i++) {
                network.add(new HiddenLayer(neuronCount, neuronCount, SEED, learningRate));
            }
            network.add(new OutputLayer(numberOfOutputs, neuronCount, SEED, learningRate));
        }

    }

    public double[] forwardPass(Image image){
        double[] outputVector = image.getImageVector();
        for(Layer layer : network){
            outputVector = layer.forwardPass(outputVector);
        }
        return outputVector;
    }

    public void backwardsPass(Image image){
        Layer layer;
        layer = network.getLast();
        layer.adjustParameters(image.getLabel());
        for(int i = network.size() - 2; i >= 0; i--){
            layer = network.get(i);
            layer.adjustParameters(network.get(i+1).getDeltas(), network.get(i+1).getWeights());
        }
    }

    public void train(List<Image> trainingData){
        double[] output;
        double loss;
        int iteration = 1;
        int guess;
        for(Image image : trainingData){
            loss = 0;
            output = forwardPass(image);
            backwardsPass(image);
            guess = 0;
            for(int i = 0; i < output.length; i++){
                if(output[0] < output[i])
                    guess = i;
            }

            for(double element : TensorUtil.loss(image.getLabel(), output))
                loss += element;
            System.out.println(
                    "(Iteration " + iteration + ")\nLoss: " + loss +
                    "\nGuess: " + guess +
                    "\nActual: " + image.getLabel() +
                    "\nFull Output: " + TensorUtil.vectToString(output) +
                    "\nImage: " + TensorUtil.vectToString(image.getImageVector()) +
                    "\n-------------------" );
            iteration++;
        }
        System.out.println("Training Complete.");
    }

}
