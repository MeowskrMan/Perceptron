package data;

public class Image {
    public  double[] imageVector;
    public int label;

    public Image(int label, double[] imageVector) {
        this.imageVector = imageVector;
        this.label = label;
    }

    public double[] getImageVector() {
        return imageVector;
    }

    public int getLabel() {
        return label;
    }
}
