package linalg;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * A class that allows simple Nd4j related operations
 *
 * @author Sebastian Acuña
 */


public class Linalg {

    /**
     * Pads a 3d matrix in the first 2 dimensions
     * @param input
     * @param pad
     * @return
     */

    public static INDArray pad(INDArray input, int pad){
        INDArray inputMatrixPadded = Nd4j.pad(input, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Nd4j.PadMode.CONSTANT);

        return inputMatrixPadded;
    }

    /**
     * Takes away padding of a 2d matrix
     * @param input
     * @param pad
     * @return matrix without padding
     */

    public static INDArray noPadMatrix(INDArray input, int pad){
        long[] shape = input.shape();
        INDArrayIndex id0 = NDArrayIndex.interval(pad, shape[0] - pad);
        INDArrayIndex id1 = NDArrayIndex.interval(pad, shape[1] - pad);
        return input.get(id0, id1);
    }
}
