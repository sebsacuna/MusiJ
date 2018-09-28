package musical;

import linalg.SVD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * A class that contains method for computing the singular values use to later
 * generate a MUSICAL image
 *
 * @author Sebastian Acuña
 *
 */

public class Singular {

    /**
     * Generates a list of singular values produced for each point in the image. Skips the
     * edges to avoid the mask to go out of the image. It is recommended to pad the image
     *
     * @param input is the image to use
     * @param mask is a square matrix representing a mask
     * @return a list of Nd4j vectors containing all the singlar values for each point in
     *         the image
     */

    public static List<INDArray> SingularValues(INDArray input, INDArray mask){

        //INDArray data = Nd4j.pad(input, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Nd4j.PadMode.CONSTANT);

        long[] input_shape = input.shape();
        long[] mask_shape = mask.shape();

        int pad = GaussianMask.calculatePadding((int)mask_shape[0]);

        int mask_size = (int)(mask_shape[0] * mask_shape[1]);

        INDArray mask_flat = mask.reshape(mask_size, 1);

        int rows = mask_size;
        int cols = (int)input_shape[2];

        List<INDArray> singularvalues = new ArrayList<>();

        int dim0_limit = (int)input_shape[0] - pad;
        int dim1_limit = (int)input_shape[1] - pad;

        SVD svd = new SVD(rows, cols);

        for(int i = pad; i < dim0_limit ; i++){
            for(int j = pad; j < dim1_limit; j++){

                INDArray roi = input.get(NDArrayIndex.interval(i-pad,i+pad+1),
                        NDArrayIndex.interval(j-pad,j+pad+1),
                        NDArrayIndex.all());

                INDArray reshaped = roi.reshape('f', rows, cols);

                INDArray masked_roi = reshaped.mulColumnVector(mask_flat);

                svd.ComputeSVD(masked_roi);

                INDArray sv = svd.getS().dup();

                singularvalues.add(sv);
            }
        }

        return singularvalues;
    }

    public static void main(String[] args) {
        /*
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        INDArray sample1 = Nd4j.create(new float[][][]{{{1,2},{3,4}},{{5,6},{7,8}}});

        System.out.println(sample1);

        System.out.println(sample1.reshape('f',4, 2));
        System.out.println(sample1.reshape('c',4, 2));
        System.out.println(sample1.transpose().reshape('c',4, 2));
        System.out.println(sample1.reshape(2, 4).transpose());
        System.out.println(sample1.reshape('f', 2, 4).transpose());

        INDArray sample2 = sample1.swapAxes(0,2);


        System.out.println(sample2);
        */
    }
}
