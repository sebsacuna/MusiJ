package musical;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A class that contains a method for creating a gaussian mask Nd4j matrix
 *
 * @author Sebastian Acuña
 */

public class GaussianMask{

    /**
     * Generates a square matrix of shape [N,N] with a 2D multivariate gaussian
     * centered in the middle. It is normalized by the maximum, which means the
     * central value is 1.0
     *
     * @param N is the size of the mask
     * @return Nd4j 2d matrix
     */

    public static INDArray getMask(int N){

        double center = Math.floor(N / 2);
        double [] means = {center, center};
        double [][] cov = {{center, 0},{0, center}};

        MultivariateNormalDistribution mnd = new MultivariateNormalDistribution(means, cov);

        float [][] gm = new float[N][N];
        double max_value = mnd.density(means);

        for(int i = 0;i < N; i++){
            for(int j = 0;j < N; j++) {
                double[] point = {i, j};
                //Symmetric, but to be clear...
                gm[j][i] = (float)(mnd.density(point) / max_value);
            }
        }

        return Nd4j.create(gm);
    }


    /**
     * Computes the padding needed when using a size of n_w
     * @param N
     * @return
     */

    public static int calculatePadding(int N){
        return (int)Math.floor(N/2.0);
    }

}
