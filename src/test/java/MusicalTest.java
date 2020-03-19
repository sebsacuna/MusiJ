import Exceptions.WrongShape;
import images.ImageReader;
import linalg.Linalg;
import linalg.SVD;
import musical.*;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;

public class MusicalTest {

    ImgPlus<FloatType> inputImgPlus;
    static ImageJ ij;

    @BeforeClass
    public static void SetupClass(){
        //Nd4j.setDataType(DataBuffer.Type.FLOAT);
        ij = new ImageJ();
    }

    @Test
    public void testSynEx1SubWindo_3_3(){

        double delta = 1e-3;

        String path = "src/test/samples/SynEx1.tiff";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }

        INDArray mask = GaussianMask.getMask(7);
        INDArray mask_flat = mask.reshape(49, 1);

        assertEquals(0.223130160148430, mask_flat.getFloat(3), delta * 0.223130160148430);

        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);

        gmatrix = gmatrix.divi(gmatrix.maxNumber().floatValue()).mulColumnVector(mask_flat);
        INDArray gm = gmatrix.transpose();
        INDArray gmmod = Nd4j.create(19600, 1);
        gm.mul(gm).sum(gmmod, 1);
        assertEquals(4.533214126048625e-05, gmatrix.getFloat(0,0), delta * 4.533214126048625e-05);

        int pad = 3;

        INDArray data = Nd4j.pad(inputMatrix, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Pad.Mode.CONSTANT, 0);;
        int i = 3 + pad;
        int j = 3 + pad;
        INDArray roi = data.get(NDArrayIndex.interval(i - pad, i + pad + 1),
                NDArrayIndex.interval(j - pad, j + pad + 1),
                NDArrayIndex.all());

        INDArray reshaped = roi.reshape('f', 49, 49);

        assertEquals(0.010193026626993, reshaped.getFloat(0,0), delta * 0.010193026626993);
        assertEquals(0.003692683299001, reshaped.getFloat(5,0), delta * 0.003692683299001);

        INDArray masked_roi = reshaped.mulColumnVector(mask_flat);

        assertEquals(0.001460636891793, masked_roi.getFloat(3, 4), delta * 0.001460636891793);

        INDArray res = Nd4j.create(19600, 49, 'f');
        INDArray bufferPS = Nd4j.create(19600, 1);
        INDArray bufferPN = Nd4j.create(19600, 1);

        SVD svd = new SVD(49,49);

        INDArray ds = Musical2.SubpixelWindow(masked_roi, gm, gmmod, res, bufferPS,bufferPN, svd, 49, 140, 0.1f, 4);

        INDArray S = svd.getS();
        INDArray U = svd.getU();

        assertEquals(6.281886667497467, S.getFloat(0), delta * 6.281886667497467);
        assertEquals(0.350726213370044, S.getFloat(1), delta * 0.350726213370044);
        assertEquals(0.272332283118305, S.getFloat(2), delta * 0.272332283118305);
        assertEquals(0.034065967261762, S.getFloat(3), delta * 0.034065967261762);

        /*
        assertEquals(-4.864120599645805e-04, U.getFloat(0,0), delta);
        assertEquals(-0.002808839674188, U.getFloat(1,1), delta);
        assertEquals(-3.863618874825693e-04, U.getFloat(4,0), delta);
        assertEquals(-0.011484860092269, U.getFloat(0,4), delta);
        */

        assertEquals(0.006271796530584, ds.getFloat(0,0), delta * 0.006271796530584);
        assertEquals(0.004360473809468, ds.getFloat(1,137), delta * 0.004360473809468);

    }

    @Test
    public void testComplexSample1_4_3(){

        double delta = 1e-3;

        String path = "src/test/samples/008_z00000001_c00000001-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }

        INDArray mask = GaussianMask.getMask(7);
        INDArray mask_flat = mask.reshape(49, 1);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);
        gmatrix = gmatrix.divi(gmatrix.maxNumber().floatValue()).mulColumnVector(mask_flat);
        INDArray gm = gmatrix.transpose();
        INDArray gmmod = Nd4j.create(19600, 1);
        gm.mul(gm).sum(gmmod, 1);
        int pad = 3;
        INDArray data = Nd4j.pad(inputMatrix, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Pad.Mode.CONSTANT, 0);;
        int i = 4 + pad;
        int j = 3 + pad;
        INDArray roi = data.get(NDArrayIndex.interval(i - pad, i + pad + 1),
                NDArrayIndex.interval(j - pad, j + pad + 1),
                NDArrayIndex.all());

        long[] data_shape = data.shape();
        long[] mask_shape = mask.shape();
        long mask_size = mask_shape[0] * mask_shape[1];

        int rows = (int)(mask_size);
        int cols = (int)data_shape[2];

        INDArray reshaped = roi.reshape('f', rows, cols);

        INDArray masked_roi = reshaped.mulColumnVector(mask_flat);

        INDArray res = Nd4j.create(19600, 49, 'f');
        INDArray bufferPS = Nd4j.create(19600, 1);
        INDArray bufferPN = Nd4j.create(19600, 1);

        SVD svd = new SVD(rows,cols);

        INDArray ds = Musical2.SubpixelWindow(masked_roi, gm, gmmod, res, bufferPS,bufferPN, svd, rows, 140, 0.1f, 4);

        INDArray S = svd.getS();
        INDArray U = svd.getU();

        assertEquals(23.882080816566269, S.getFloat(0), delta * 23.882080816566269);
        assertEquals(0.706501436493783, S.getFloat(1), delta * 0.706501436493783);
        assertEquals(0.607838855070700, S.getFloat(2), delta * 0.607838855070700);
        assertEquals(0.565862799350835, S.getFloat(3), delta * 0.565862799350835);

        /*
        assertEquals(-4.864120599645805e-04, U.getFloat(0,0), delta);
        assertEquals(-0.002808839674188, U.getFloat(1,1), delta);
        assertEquals(-3.863618874825693e-04, U.getFloat(4,0), delta);
        assertEquals(-0.011484860092269, U.getFloat(0,4), delta);
        */

        assertEquals(0.464386043642758, ds.getFloat(0,0), delta * 0.464386043642758);
        assertEquals(2.975258157208491e+04, ds.getFloat(16,71), delta * 2.975258157208491e+04);
        assertEquals(88.539234754905420, ds.getFloat(54,4), delta * 88.539234754905420);

    }

    @Test
    public void testComplexSample1_22_48(){

        double delta = 1e-3;

        String path = "src/test/samples/008_z00000001_c00000001-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        INDArray mask = GaussianMask.getMask(7);
        INDArray mask_flat = mask.reshape(49, 1);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);
        gmatrix = gmatrix.divi(gmatrix.maxNumber().floatValue()).mulColumnVector(mask_flat);
        INDArray gm = gmatrix.transpose();
        INDArray gmmod = Nd4j.create(19600, 1);
        gm.mul(gm).sum(gmmod, 1);
        int pad = 3;
        INDArray data = Nd4j.pad(inputMatrix, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Pad.Mode.CONSTANT, 0);;
        int i = 22 + pad;
        int j = 48 + pad;
        INDArray roi = data.get(NDArrayIndex.interval(i - pad, i + pad + 1),
                NDArrayIndex.interval(j - pad, j + pad + 1),
                NDArrayIndex.all());

        long[] data_shape = data.shape();
        long[] mask_shape = mask.shape();
        long mask_size = mask_shape[0] * mask_shape[1];

        int rows = (int)(mask_size);
        int cols = (int)data_shape[2];

        INDArray reshaped = roi.reshape('f', rows, cols);

        INDArray masked_roi = reshaped.mulColumnVector(mask_flat);

        INDArray res = Nd4j.create(19600, 49, 'f');
        INDArray bufferPS = Nd4j.create(19600, 1);
        INDArray bufferPN = Nd4j.create(19600, 1);

        SVD svd = new SVD(rows,cols);

        INDArray ds = Musical2.SubpixelWindow(masked_roi, gm, gmmod, res, bufferPS,bufferPN, svd, rows, 140, 0.1f, 4);

        INDArray S = svd.getS();
        INDArray U = svd.getU();

        assertEquals(13.527999914642376, S.getFloat(0), delta * 13.527999914642376);
        assertEquals(0.372223670368209, S.getFloat(1), delta * 0.372223670368209);
        assertEquals(0.341677844600507, S.getFloat(2), delta * 0.341677844600507);
        assertEquals(0.325022761464882, S.getFloat(3), delta * 0.325022761464882);

        /*
        assertEquals(-4.864120599645805e-04, U.getFloat(0,0), delta);
        assertEquals(-0.002808839674188, U.getFloat(1,1), delta);
        assertEquals(-3.863618874825693e-04, U.getFloat(4,0), delta);
        assertEquals(-0.011484860092269, U.getFloat(0,4), delta);
        */

        assertEquals(0.084392329170940, ds.getFloat(0,0), delta * 0.084392329170940);
        assertEquals(45.499944399604956, ds.getFloat(16,71), delta * 45.499944399604956);
        assertEquals(5.651544762815767, ds.getFloat(54,4), delta * 5.651544762815767);

    }



    @Test
    public void testComplexSample1result(){

        double delta = 1e-3;

        String path = "src/test/samples/008_z00000001_c00000001-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        int pad = GaussianMask.calculatePadding(7);
        INDArray inputMatrixPadded = Linalg.pad(inputMatrix, pad);

        INDArray mask = GaussianMask.getMask(7);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);

        Status s = new Status(0, 100);
        INDArray musical_result = Musical2.WrapperGetMusical(inputMatrixPadded, mask,gmatrix, 20,-1,(float)4, s);
        musical_result = Linalg.noPadMatrix(musical_result, 20 * 3);
        assertEquals(8.384156987929458e+03, musical_result.getFloat(530, 499), delta * 8.384156987929458e+03);
        assertEquals(9.727832055731615e+04, musical_result.getFloat(150, 600), delta * 9.727832055731615e+04);
    }

    @Test
    public void testThreadedComplexSample1result(){

        double delta = 1e-3;

        String path = "src/test/samples/008_z00000001_c00000001-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }

        INDArray mask = GaussianMask.getMask(7);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);

        INDArray inputMatrixPadded = Linalg.pad(inputMatrix, 3);
        Status s = new Status(0, 100);
        INDArray musical_result = Musical2.WrapperGetMusical(inputMatrixPadded, mask, gmatrix, 20,-1,(float)4, 4, s);

        musical_result = Linalg.noPadMatrix(musical_result, 20 * 3);
        assertEquals(9.727832055731615e+04, musical_result.getFloat(150, 600), delta * 9.727832055731615e+04);
        assertEquals(8.384156987929458e+03, musical_result.getFloat(530, 499), delta * 8.384156987929458e+03);
    }

    @Test
    public void testThreadEqualitySynEx2(){

        double delta = 1e-3;

        //String path = "src/test/samples/SynEx2.tiff";
        String path = "src/test/samples/008_z00000001_c00000001-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        int pad = GaussianMask.calculatePadding(7);
        INDArray inputMatrixPadded = Linalg.pad(inputMatrix, pad);

        INDArray mask = GaussianMask.getMask(7);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49,7, 0.065f, 20);
        Status s = new Status(0, 100);
        INDArray musical_result_thread = Musical2.WrapperGetMusical(inputMatrixPadded, mask, gmatrix, 20,-1,(float)4, 2, s);

        //musical_result_thread = Musical.noPadMatrix(musical_result_thread, 20 * 3);

        INDArray musical_result_single = Musical2.WrapperGetMusical(inputMatrixPadded, mask,gmatrix, 20,-1,(float)4, s);
        //musical_result_single = Musical.noPadMatrix(musical_result_single, 20 * 3);

        for(int i = 0; i < musical_result_single.shape()[0]; i++){
            for(int j = 0; j < musical_result_single.shape()[1]; j++){
                float a = musical_result_single.getFloat(i, j);
                float b = musical_result_thread.getFloat(i, j);
                if(b != 0) {
                    if (Math.abs(a - b) > Math.abs(delta * b)) {
                        fail();
                    }
                }
                else{
                    if (Math.abs(a - b) > Math.abs(delta)){
                        fail();
                    }
                }
            }
        }
    }

    @Test
    public void testThreadEqualityComplex(){

        double delta = 1e-3;

        //String path = "src/test/samples/SynEx2.tiff";
        String path = "src/test/samples/C3-20180719_H9c2_1_dsRed-mito_KDEL_mC1to2000-20m_T23_1516_m_046C_608nm-1.tif";
        File f = new File(path);
        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        int pad = GaussianMask.calculatePadding(7);
        INDArray inputMatrixPadded = Linalg.pad(inputMatrix, pad);

        INDArray mask = GaussianMask.getMask(7);
        double imagePixSize = Gmatrix.calculateImagePixelSize(80,1);
        INDArray gmatrix = Gmatrix.GenerateGMatrix(583, 1.42,7, imagePixSize, 20);

        Status s = new Status(0, 100);

        INDArray musical_result_single = Musical2.WrapperGetMusical(inputMatrixPadded, mask,gmatrix, 20,-1,(float)4, s);
        musical_result_single = Linalg.noPadMatrix(musical_result_single, 20 * 3);
        INDArray sample_column = musical_result_single.get(NDArrayIndex.interval(669,680),NDArrayIndex.point(739));

        double[] true_values = {
                3.025861378355313,
                3.075463476589118,
                3.132939984283447,
                3.198146469277435,
                3.270813323841141,
                3.350515313298810,
                3.436642211584097,
                3.528374355621328,
                3.624667575871901,
                3.724252140102316,
                3.825649897157724
        };

        for(int i = 0; i < true_values.length; i++){
            assertEquals(true_values[i], sample_column.getFloat(i),delta * true_values[i]);
        }
    }
}

