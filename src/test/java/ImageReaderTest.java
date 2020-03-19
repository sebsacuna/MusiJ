import Exceptions.WrongShape;
import images.ImageReader;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.fail;

public class ImageReaderTest {

    ImgPlus<FloatType> inputImgPlus;
    static ImageJ ij;

    @BeforeClass
    public static void SetupClass(){
        //Nd4j.setDataType(DataBuffer.Type.FLOAT);
        ij = new ImageJ();
    }

    @Before
    public void Setup(){
        String path = "src/test/samples/SynEx1.tiff";
        File f = new File(path);

        try{
            Dataset d = ij.scifio().datasetIO().open(f.getPath());
            inputImgPlus = (ImgPlus<FloatType>)d.getImgPlus();
        }
        catch (IOException exc){
            System.out.println("Problem reading the file");
        }
    }

    @Test
    public void testValuesOfImage(){
        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, false);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        assertEquals(668.0, inputMatrix.getFloat(new int[]{0,0,0}), 1e-6);
        assertEquals(15970, inputMatrix.getFloat(new int[]{4,4,0}), 1e-6);
        assertEquals(228.0, inputMatrix.getFloat(new int[]{0,4,0}), 1e-6);
        assertEquals(320.0, inputMatrix.getFloat(new int[]{6,4,4}), 1e-6);
        assertEquals(841.0, inputMatrix.getFloat(new int[]{6,6,4}), 1e-6);
    }

    @Test
    public void testValuesOfImageNormMax(){
        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            wrongShape.printStackTrace();
            fail();
        }
        assertEquals(0.002105745021744, inputMatrix.getFloat(new int[]{0,1,0}), 1e-9);
        assertEquals(0.234393835355154, inputMatrix.getFloat(new int[]{4,4,2}), 1e-9);
        assertEquals(0.003479056992447, inputMatrix.getFloat(new int[]{0,4,0}), 1e-9);
        assertEquals(0.004882887006943, inputMatrix.getFloat(new int[]{6,4,4}), 1e-9);
        assertEquals(0.012832837415122, inputMatrix.getFloat(new int[]{6,6,4}), 1e-9);
    }
}
