package tensor;

public class TensorUtil {

    public static double[] multiply(double[][] matrix, double[] vector){
        int cols = matrix[0].length;
        int rows = matrix.length;
        double[] output = new double[rows];
        for (double[] doubles : matrix) {
            for (int j = 0; j < rows; j++) {
                output[j] = vector[j] * doubles[j];
            }
        }
        return output;
    }

    public static double[] add(double[] vector1, double[] vector2){
        int len = vector1.length;
        double[] output = new double[len];
        for(int i = 0; i < len; i++)
            output[i] = vector1[i] + vector2[i];
        return output;
    }

    public static double[][] multiply(double[][] matrix, double scalar){
        double[][] output = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                output[i][j] = scalar*matrix[i][j];
            }
        }
        return output;
    }

    public static double[][] add(double[][] matrix, double[][] matrix2){
        double[][] output = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                output[i][j] = matrix2[i][j]*matrix[i][j];
            }
        }
        return output;
    }

    public static double[] reLU(double[] vector){
        int len = vector.length;
        double[] output = new double[len];
        for(int i = 0; i < len; i++)
            output[i] = Math.max(0, vector[i]);
        return output;
    }

    public static double loss(double[] actual, double[] predicted){
        double output = 0;
        for(int i = 0; i < predicted.length; i++){
            output += 0.5*Math.pow(actual[i] - predicted[i], 2.0);
        }
        return output;
    }

    public static double[] derivativeLoss(double[] actual, double[] predicted){
        double[] output = new double[actual.length];
        for(int i = 0; i < predicted.length; i++){
            output[i] = actual[i] - predicted[i];
        }
        return output;
    }

    public static double[] sigmoid(double[] vector){
        double[] output = new double[vector.length];
        for(int i = 0; i < vector.length; i++){
            output[i] = 1.0/(1+Math.exp(vector[i]*-1.0));
        }
        return output;
    }


}
