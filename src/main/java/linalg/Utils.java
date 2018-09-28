package linalg;

public class Utils {

    public static float[] log10array(float[] x){
        float[] res = new float[x.length];
        for(int i = 0;i<res.length;i++){
            res[i] = (float)Math.log10(x[i]);
        }
        return res;
    }

    public static double[] invertArray(double[] x){
        double[] x_inv = new double[x.length];
        for(int i=0;i<x.length;i++){
            x_inv[i]=x[x.length-1-i];
        }
        return x_inv;
    }
}
