package layer;

import tensor.TensorUtil;

import java.util.Random;

public class PerceptronLayer {
    private double[][] weights;
    private double[] biases;
    private int neuronCount;
    private double[] lastWeightedSum;
    private double[] lastInput;

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


    public double[] forwardPass(double[] input){
        this.lastInput = input;
        this.lastWeightedSum = TensorUtil.add(TensorUtil.multiply(weights, input), biases);
        return TensorUtil.reLU(this.lastWeightedSum);
    }

    public double[] backwardPass(double[] dLdA){
        double[] dLdX = new double[weights[0].length];
        double dLdW;
        double dAdZ;
        double dZdW;
        double dZdX;
        for(int i = 0; i < weights[0].length; i++){
            double dLdX_Sum = 0;
            for(int j = 0; j< weights.length; j++){
                dAdZ = derivativeRelu(lastWeightedSum[j]);
                dZdW = lastInput[j];
                dLdW = dLdA[j]*dAdZ*dZdW;
                dZdX = weights[i][j];
                weights[i][j] -= 0.01*dLdW;
                dLdX_Sum += dLdA[j]*dAdZ*dZdX;
            }
            dLdX[i] = dLdX_Sum;
        }
        return dLdX;
    }

    public static double derivativeRelu(double scalar){
        if(scalar <= 0) return 1;
        else return 0.01;
    }
}
