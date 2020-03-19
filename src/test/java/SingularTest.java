import Exceptions.WrongShape;
import images.ImageReader;
import musical.GaussianMask;
import musical.Gmatrix;
import musical.Singular;
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

import java.io.File;
import java.io.IOException;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;

public class SingularTest {

    ImgPlus<FloatType> inputImgPlus;
    static ImageJ ij;

    @BeforeClass
    public static void SetupClass(){
        //Nd4j.setDataType(DataBuffer.Type.FLOAT);
        ij = new ImageJ();
    }

    @Test
    public void testSynEx1(){
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
        int pad = GaussianMask.calculatePadding(7);
        INDArray inputMatrixPadded = Nd4j.pad(inputMatrix, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Pad.Mode.CONSTANT, 0);

        INDArray mask = GaussianMask.getMask(7);

        List<INDArray> svlist = Singular.SingularValues(inputMatrixPadded, mask);

        assertEquals(0.732475109597127, svlist.get(0).getFloat(0), 1e-6);
        assertEquals(0.050802486246609, svlist.get(0).getFloat(1), 1e-6);
        assertEquals(0.032795336481166, svlist.get(0).getFloat(2), 1e-6);
        assertEquals(0.005281097442818, svlist.get(0).getFloat(3), 1e-6);
        assertEquals(0.000033011759330, svlist.get(0).getFloat(4), 1e-6);
        assertEquals(0.000025627841117, svlist.get(0).getFloat(5), 1e-6);
        assertEquals(0.000004169249082, svlist.get(0).getFloat(14), 1e-6);
        assertEquals(0.000002051965198, svlist.get(0).getFloat(15), 1e-6);
    }

    @Test
    public void testC3_20180719(){
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
        double imagePixelSize = Gmatrix.calculateImagePixelSize(80,1);
        int n_w = Gmatrix.calculateWindowSize(583, 1.42, imagePixelSize );
        int pad = GaussianMask.calculatePadding(n_w);
        INDArray inputMatrixPadded = Nd4j.pad(inputMatrix, new int[][]{{pad,pad}, {pad,pad}, {0,0}}, Pad.Mode.CONSTANT, 0);

        INDArray mask = GaussianMask.getMask(n_w);

        List<INDArray> svlist = Singular.SingularValues(inputMatrixPadded, mask);

        int x = 0;
        int y = 0;

        INDArray s_0_0 = svlist.get((int)(inputMatrix.shape()[0]*y + x));

        x = 49;
        y = 23;

        INDArray s_23_49 = svlist.get((int)(inputMatrix.shape()[1]*y + x));

        double delta = 1e-4;

        double[] singularvalues0_0 = {2.98019580306311,
                0.123591193600511,
                0.103724212726300,
                0.0820397142141337,
                0.0765393153867630,
                0.0492736796030150,
                0.0425854145222651,
                0.0398663504192242};

        double[] singularvalues23_49 = {4.22878025423293,
                0.135234291905111,
                0.106254603273987,
                0.0972105724999428,
                0.0933519541927444,
                0.0869017598156039,
                0.0841429639703534};

        for(int i = 0; i < singularvalues0_0.length; i++){
            assertEquals(singularvalues0_0[i], s_0_0.getFloat(i), delta * singularvalues0_0[i]);
        }

        for(int i = 0; i < singularvalues23_49.length; i++){
            assertEquals(singularvalues23_49[i], s_23_49.getFloat(i), delta * singularvalues23_49[i]);
        }
    }
}

