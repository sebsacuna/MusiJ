package linalg;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A reusable class that allows SVD computation of fixed size matrices
 *
 * @author Sebastian Acuña
 */

public class SVD {
    INDArray U;
    INDArray S;
    INDArray V;

    public SVD(int rows, int columns){
        // Check if order generates problem. It used to be the case that only worked with order 'f'
        S = Nd4j.create(rows);
        U = Nd4j.create(rows, rows);
        V = Nd4j.create(columns, columns);
    }

    public void ComputeSVD(INDArray data){
        //data.setOrder('c');
        Nd4j.getBlasWrapper().lapack().gesvd(data,S,U,V);
    }

    public INDArray getS() {
        return S;
    }

    public INDArray getU() {
        return U;
    }

    public INDArray getV() {
        return V;
    }

    /**
     * Example of using SVD. All the calls are made in prints
     * @param args
     */

    public static void main(String[] args) {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        INDArray sample1 = Nd4j.create(new float[][]{{1,2},{3,4}});

        printSVD(sample1);

        INDArray sample2 = Nd4j.create(new float[][]{{1,2,3},{4,5,6}});

        printSVD(sample2);
        sample2 = Nd4j.create(new float[][]{{1,2,3},{4,5,6}});
        printSVD(sample2.transpose());

    }

    public static void printSVD(INDArray data){
        System.out.println("data:");
        System.out.println(data);
        long[] shape = data.shape(); // 1.0.0-beta2
        //int[] shape = data.shape(); // 1.0.0-alpha
        SVD svd = new SVD((int)shape[0],(int)shape[1]);

        svd.ComputeSVD(data);

        System.out.println("S");
        System.out.println(svd.getS());
        System.out.println(("U"));
        System.out.println(svd.getU());
        System.out.println(("V"));
        System.out.println(svd.getV());

        System.out.println("Check data again:");
        System.out.println(data);

    }
}
