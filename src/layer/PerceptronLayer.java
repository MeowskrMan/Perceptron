package layer;

import tensor.TensorUtil;

import java.util.Random;

public class PerceptronLayer {
    private double[][] weights;
    private double[] biases;
    private int neuronCount;

    public PerceptronLayer(int neurons, int inputCount, long SEED){
        this.weights = new double[neurons][inputCount];
        this.biases = new double[neurons];
        this.neuronCount = neurons;
        Random random = new Random(SEED);
        for(double[] doubles : this.weights)
            for(int i = 0; i < doubles.length; i++)
                doubles[i] = random.nextGaussian();
        for(int i = 0; i < this.biases.length; i++)
            this.biases[i] = random.nextGaussian();
    }


    public double[] getOutput(double[] input){
        return TensorUtil.reLU(TensorUtil.add(TensorUtil.multiply(weights, input), biases));
    }

}
