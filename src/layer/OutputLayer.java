package layer;

import tensor.TensorUtil;

public class OutputLayer extends Layer {
    public OutputLayer(int neurons, int inputCount, long SEED, double learningRate){
        super(neurons, inputCount, SEED, learningRate);
    }

    @Override
    public void setDeltas(double actual){
        double[] output = new double[this.biases.length];
        double[] losses = TensorUtil.derivativeLoss(actual, this.lastSigmoid);
        double[] sigmoid = TensorUtil.derivativeSigmoid(this.lastSigmoid);
        for(int i = 0; i < this.biases.length; i++){
            output[i] = losses[i] * sigmoid[i];
        }
        this.deltas = output;
    }

    @Override
    public void adjustParameters(double actual){
        this.setDeltas(actual);
        adjustHelper();
    }

    @Override
    public void setDeltas(double[] nextLayerDeltas, double[][] nextLayerWeights) {

    }

    @Override
    public void adjustParameters(double[] nextLayerDeltas, double[][] nextLayerWeights) {

    }


}
