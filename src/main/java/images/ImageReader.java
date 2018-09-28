package images;

import Exceptions.WrongShape;
import net.imagej.ImgPlus;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A class that allows convert images from ImgLib2 to Nd4j matrices
 *
 * @author Sebastian Acuña
 */

public class ImageReader {

    /**
     * Converts an Img into an 3d Nd4j. It assumes the image has 3 dimensions.
     *
     * @param img img to convert to matrix
     * @param normMax if true, the 3d tensor will be normalized by its maximum value
     * @param <T> data type of the image
     * @return a 3d Nd4j tensor
     */

    public static <T extends RealType<T>> INDArray ImgToMatrix(Img<T> img, boolean normMax) throws WrongShape{
        int dims = img.numDimensions();

        if (dims != 3){
            throw new WrongShape("The image must be a single channel stack");
        }

        // Accessor to the image. We use RandomAccess to have control over the region
        RandomAccess<T> r = img.randomAccess();
        final int n = img.numDimensions();
        final long[] d = new long[n];
        // ImgLib2 uses the order [x,y..] so attention
        img.max(d);

        // Create empty IMatrix
        //INDArray result = Nd4j.zeros(new long[]{d[1]+1, d[0] + 1, d[2] + 1});
        INDArray result = Nd4j.zeros(new int[]{(int)d[1]+1, (int)d[0] + 1, (int)d[2] + 1});
        for(int i = 0; i <= d[0]; i++){
            for(int j = 0; j <= d[1]; j++){
                for(int k = 0; k <= d[2]; k++){
                    long[] position = {i, j, k};
                    r.setPosition(position);
                    T t = r.get();
                    float pixel_value = t.getRealFloat();
                    result.putScalar(new int[]{j, i, k}, pixel_value);
                }
            }
        }

        if(normMax){
            result.divi(result.maxNumber().floatValue());
        }

        return result;
    }

    /**
     * Converts an array into ImgPlus. Check if can be the superclass Img
     *
     * @param matrix is a 2d Nd4j matrix tha contains the image.
     * @param type data object representing the type T
     * @param name name of the image
     * @param <T> is the type used to save the image. Must be an Integer type
     * @return an ImgPlus object
     */

    public static <T extends IntegerType<T> & NativeType< T >> ImgPlus MatrixToImg(INDArray matrix, T type, String name){
        ImgFactory<T> imgFactory = new ArrayImgFactory(type);

        long[] matrix_shape = matrix.shape();

        ImgPlus<T> result = new ImgPlus <>(imgFactory.create(new long[] {matrix_shape[1], matrix_shape[0]}, type), name);

        int bit_count = type.getBitsPerPixel();
        float max_value = (float)(Math.pow(2, bit_count) - 1);

        matrix.divi(matrix.maxNumber().floatValue()).muli(max_value);

        RandomAccess<T> r = result.randomAccess();
        // TODO: check order
        for(int i = 0; i < matrix_shape[1]; i++){
            for(int j = 0; j < matrix_shape[0]; j++){
                long[] position = {i, j};
                r.setPosition(position);
                T t = r.get();
                t.setInteger((int)matrix.getFloat(j,i));
            }
        }

        return result;
    }

    public static <T extends RealType<T>> INDArray ImgToMatrix2D(Img<T> img, boolean normMax) throws WrongShape{
        int dims = img.numDimensions();

        if (dims != 2){
            throw new WrongShape("The image must have 3 dimensions");
        }

        // Accessor to the image. We use RandomAccess to have control over the region
        RandomAccess<T> r = img.randomAccess();
        final int n = img.numDimensions();
        final long[] d = new long[n];
        // ImgLib2 uses the order [x,y..] so attention
        img.max(d);

        // Create empty IMatrix
        //INDArray result = Nd4j.zeros(new long[]{d[1]+1, d[0] + 1, d[2] + 1});
        INDArray result = Nd4j.zeros(new int[]{(int)d[1]+1, (int)d[0] + 1});
        for(int i = 0; i <= d[0]; i++){
            for(int j = 0; j <= d[1]; j++){
                long[] position = {i, j};
                r.setPosition(position);
                T t = r.get();
                float pixel_value = t.getRealFloat();
                result.putScalar(new int[]{j, i}, pixel_value);
            }
        }

        if(normMax){
            result.divi(result.maxNumber().floatValue());
        }

        return result;
    }

}
