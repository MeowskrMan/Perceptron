package layer;

import tensor.TensorUtil;
import java.util.Random;

public abstract class Layer {
    double[][] weights;
    double[] biases;
    private double[] lastZ;
    double[] lastInput;
    double[] lastSigmoid;
    double[] deltas;
    double learningRate;

    public Layer(int neurons, int inputCount, long SEED, double learningRate){
        this.weights = new double[neurons][inputCount];
        this.biases = new double[neurons];
        this.learningRate = learningRate;
        Random random = new Random(SEED);

        // Generating Gaussian Weights.
        for(double[] doubles : this.weights)
            for(int i = 0; i < doubles.length; i++)
                doubles[i] = random.nextGaussian();

        // Generating Gaussian Biases.
        for(int i = 0; i < this.biases.length; i++)
            this.biases[i] = random.nextGaussian();
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getDeltas() {
        return deltas;
    }

    public double[] forwardPass(double[] input){
        this.lastInput = input;
        this.lastZ = TensorUtil.add(TensorUtil.multiply(this.weights, input), this.biases);
        this.lastSigmoid = TensorUtil.sigmoid(this.lastZ);
        return this.lastSigmoid;
    }

    public void adjustHelper(){
        for(int i = 0; i < this.biases.length; i++) {
            this.biases[i] -= this.learningRate * this.deltas[i];
            for (int j = 0; j < this.weights[0].length; j++) {
                this.weights[i][j] -= this.learningRate*deltas[i]*this.lastInput[i];
            }
        }
    }

    // Abstract methods for polymorphism
    public abstract void setDeltas(double actual);
    public abstract void adjustParameters(double actual);
    public abstract void setDeltas(double[] nextLayerDeltas, double[][] nextLayerWeights);
    public abstract void adjustParameters(double[] nextLayerDeltas, double[][] nextLayerWeights);


}
