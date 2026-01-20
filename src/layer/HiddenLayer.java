package layer;

import tensor.TensorUtil;

public class HiddenLayer extends Layer {
    public HiddenLayer(int neurons, int inputCount, long SEED, double learningRate){
        super(neurons, inputCount, SEED, learningRate);
    }

    @Override
    public void setDeltas(double actual) {

    }

    @Override
    public void adjustParameters(double actual) {

    }

    @Override
    public void setDeltas(double[] nextLayerDeltas, double[][] nextLayerWeights){
        double[] output = new double[this.biases.length];
        double[] sigmoid = TensorUtil.derivativeSigmoid(this.lastSigmoid);
        double sum;
        for(int i = 0; i < this.biases.length; i++){
            sum = 0;
            for(int j = 0; j < nextLayerDeltas.length; j++){
                sum += nextLayerDeltas[j] * nextLayerWeights[j][i] * sigmoid[i];
            }
            output[i] = sum;
        }
        this.deltas = output;
    }

    @Override
    public void adjustParameters(double[] nextLayerDeltas, double[][] nextLayerWeights){
        this.setDeltas(nextLayerDeltas, nextLayerWeights);
        adjustHelper();
    }


}
