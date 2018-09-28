package musical;

import linalg.SVD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * A class that contain static methods for generating an image using MUSICAL
 *
 * @author Sebastian Acuña
 */

public class Musical {

    /**
     * Generates musical matrix with the sumation of all subwindows and a count of windows
     * per pixel
     * @param input image to work with. Assumes it is 3d
     * @param mask 2d mask. Should be normalized
     * @param gmatrix 2d matrix
     * @param subpixels
     * @param threshold
     * @param alpha
     * @return a list of 2 matrices
     */

    public static List<INDArray> GetMusicalMatrices(INDArray input, INDArray mask, INDArray gmatrix, int subpixels, float threshold, float alpha ){

        //INDArray data = Nd4j.pad(input, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Nd4j.PadMode.CONSTANT);

        long[] input_shape = input.shape();
        long[] mask_shape = mask.shape();

        float threshold10 = (float) Math.pow(10, threshold);

        int pad = GaussianMask.calculatePadding((int)mask_shape[0]);

        long[] gmatrix_shape = gmatrix.shape();

        int mask_size = (int)(mask_shape[0] * mask_shape[1]);

        INDArray mask_flat = mask.reshape(mask_size, 1);

        INDArray gm = gmatrix.div(gmatrix.maxNumber().floatValue()).mulColumnVector(mask_flat);

        int rows = (int)(mask_size);
        int cols = (int)input_shape[2];

        int msize = (int)(mask_shape[0] * subpixels);

        int dim0_limit = (int)input_shape[0] - pad;
        int dim1_limit = (int)input_shape[1] - pad;

        // It is faster for some reason to use 'f' order
        INDArray res = Nd4j.create(rows, gmatrix_shape[1], 'f');
        // Buffers
        INDArray bufferPS = Nd4j.create(1, gmatrix_shape[1]);
        INDArray bufferPN = Nd4j.create(1, gmatrix_shape[1]);
        //INDArray bufferRoi = Nd4j.cre

        SVD svd = new SVD(rows, cols);

        INDArray musical_result = Nd4j.zeros(input_shape[0] * subpixels, input_shape[1] * subpixels);
        INDArray musical_count = Nd4j.zeros(input_shape[0] * subpixels, input_shape[1] * subpixels);
        // TODO: check order
        long progress = 0;

        for(int i = pad; i < dim0_limit ; i++) {
            for (int j = pad; j < dim1_limit; j++) {
                INDArray roi = input.get(NDArrayIndex.interval(i - pad, i + pad + 1),
                        NDArrayIndex.interval(j - pad, j + pad + 1),
                        NDArrayIndex.all());
                INDArray reshaped_roi = roi.reshape('f', rows, cols);
                INDArray masked_roi = reshaped_roi.mulColumnVector(mask_flat);

                INDArray ds = SubpixelWindow(masked_roi, gm, res, bufferPS, bufferPN, svd, rows, msize, threshold10, alpha );

                int x_low = subpixels * (j - pad);
                int x_high = subpixels * (j + pad + 1);
                int y_low = subpixels * (i - pad);
                int y_high = subpixels * (i + pad + 1);

                INDArrayIndex iy = NDArrayIndex.interval(y_low, y_high);
                INDArrayIndex ix = NDArrayIndex.interval(x_low, x_high);
                musical_result.get(iy, ix).addi(ds);
                musical_count.get(iy, ix).addi(1);

            }
        }

        long musical_pad = pad * subpixels;
        long[] musical_shape = musical_result.shape();

        //INDArrayIndex id0 = NDArrayIndex.interval(musical_pad, musical_shape[0] - musical_pad);
        //INDArrayIndex id1 = NDArrayIndex.interval(musical_pad, musical_shape[1] - musical_pad);

        //INDArray musical_final = musical_result.divi(musical_count);
        List<INDArray> result = new ArrayList<>();
        result.add(musical_result);
        result.add(musical_count);
        return result;
        //return musical_final.get(id0, id1);
        //return musical_result.divi(musical_count);
        //return musical_result;
    }

    /**
     * Computes a windows on the given region.
     *
     * @param roi region of interest
     * @param gmatrix assumes it is normalized by the maximum
     * @param res buffer to store matrix multiplication. Shape must be [rows, gmatrix.shape[1]]
     * @param bufferPS buffer to store summation of shape [1, gmatrix.shape[1]]
     * @param bufferPN buffer to store summation of shape [1, gmatrix.shape[1]]
     * @param svd object that computes SVD. Should have initialized according to roi
     * @param rows should be equal to roi.shape[0]
     * @param msize size of the resulting window, so result will have shape [msize, msize]
     * @param threshold10 value of 10^threshold
     * @param alpha
     * @return a patch of the region in super resolution
     */

    public static INDArray SubpixelWindow(INDArray roi, INDArray gmatrix, INDArray res, INDArray bufferPS, INDArray bufferPN, SVD svd,  int rows, int msize, float threshold10, float alpha){

        svd.ComputeSVD(roi);
        INDArray sv = svd.getS();

        int m = 0;
        for(int i = 0; i<sv.length(); i++){
            m++;
            if(sv.getDouble(i)<threshold10){
                break;
            }
        }
        if(m>=sv.length()||m<=0){
            m = (int)sv.length()-3;
        }

        INDArray U = svd.getU();

        U.transpose().mmul(gmatrix, res);
        //INDArray u_mix = U.mmul(mapping_test);
        res.muli(res);
        //INDArray d_PS = res.get(NDArrayIndex.interval(0,m),NDArrayIndex.all()).sum(0);
        res.get(NDArrayIndex.interval(0,m),NDArrayIndex.all()).sum(bufferPS,0);
        //INDArray d_PN = res.get(NDArrayIndex.interval(m,rows),NDArrayIndex.all()).sum(0);
        res.get(NDArrayIndex.interval(m,rows),NDArrayIndex.all()).sum(bufferPN,0);
        //INDArray ps = Transforms.pow(d_PS.divi(d_PN),alpha/2,false);
        INDArray ps = Transforms.pow(bufferPS.divi(bufferPN),alpha/2,false);

        Transforms.reverse(ps,false);

        return ps.reshape('f', msize, msize);
    }



    /**
     * Apply MUSICAL to an image by dividing it in vertical segments. Assumes the matrix is padded
     *
     * @param input
     * @param mask
     * @param gmatrix
     * @param subpixels
     * @param threshold
     * @param alpha
     * @param cores
     * @return
     */

    public static List<INDArray> GetMusicalMatricesThreaded(INDArray input, INDArray mask, INDArray gmatrix, int subpixels, float threshold, float alpha, int cores ){

        int pad = GaussianMask.calculatePadding((int)mask.shape()[0]);
        //INDArray inputMatrixPadded = Linalg.pad(input, pad);

        //long[] input_padded_shape = inputMatrixPadded.shape();
        long[] input_shape = input.shape();

        INDArray musicalMatrixFinal = Nd4j.zeros(input_shape[0] * subpixels, input_shape[1] * subpixels);
        INDArray musicalMatrixCount = Nd4j.zeros(input_shape[0] * subpixels, input_shape[1] * subpixels);

        long height = input.shape()[0];

        INDArray lines_cpu = Nd4j.linspace(pad,height - pad, cores + 1);

        ExecutorService es = Executors.newFixedThreadPool(cores);

        List<Future<List<INDArray>>> futureList = new ArrayList<>();

        for(int i = 0; i < lines_cpu.length() - 1; i++) {
            final int j = i;
            Callable<List<INDArray>> callable = () -> {

                List<INDArray> mus_result_partial = Musical.GetMusicalMatrices(
                        input.get(NDArrayIndex.interval(lines_cpu.getInt(j) - pad, lines_cpu.getInt(j + 1) + pad),
                                NDArrayIndex.all(), NDArrayIndex.all()), mask, gmatrix, subpixels,threshold,(float)alpha);
                return mus_result_partial;
            };
            futureList.add(es.submit(callable));
        }

        for(int i = 0; i < futureList.size(); i++) {
            try {
                List<INDArray> result = futureList.get(i).get();
                int init = (lines_cpu.getInt(i) - pad) * subpixels;
                int end = (lines_cpu.getInt(i + 1) + pad) * subpixels; // is there a 1 missing?

                musicalMatrixFinal.get(NDArrayIndex.interval(init, end), NDArrayIndex.all()).addi(result.get(0));

                musicalMatrixCount.get(NDArrayIndex.interval(init, end), NDArrayIndex.all()).addi(result.get(1));
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }

        es.shutdown();

        List<INDArray> result = new ArrayList<>();
        result.add(musicalMatrixFinal);
        result.add(musicalMatrixCount);

        return result;

        //return musicalMatrixFinal;
    }

    public static INDArray WrapperGetMusical(INDArray input, INDArray mask, INDArray gmatrix, int subpixels, float threshold, float alpha){

        List<INDArray> results = GetMusicalMatrices(input, mask, gmatrix, subpixels,threshold,alpha);

        return(results.get(0).divi(results.get(1)));
    }

    public static INDArray WrapperGetMusical(INDArray input, INDArray mask, INDArray gmatrix, int subpixels, float threshold, float alpha, int cores ){

        List<INDArray> results = GetMusicalMatricesThreaded(input, mask, gmatrix, subpixels,threshold,alpha, cores);

        return(results.get(0).divi(results.get(1)));
    }
}
