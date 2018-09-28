import linalg.Point2D;
import linalg.Utils;
import musical.Gmatrix;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class GmatrixTest {

    @Test
    public void testGenerateListOfPoints20(){
        double[] x = Gmatrix.generateCoordinates(7,0.065,20);
        ArrayList<Point2D> lop = Gmatrix.generateListOfPoints(x);
        Point2D p1 = new Point2D(-0.225875,-0.225875);
        Point2D p50 = new Point2D(-0.225875,-0.066625);
        Point2D p11200 = new Point2D(0.030875,0.225875);

        assertTrue(lop.get(0).equals(p1));
        assertTrue(lop.get(49).equals(p50));
        assertTrue(lop.get(11199).equals(p11200));
    }

    @Test
    public void testGenerateCoordSub1(){
        double[] x = Gmatrix.generateCoordinates(7,0.065,1);
        assertTrue(x[0] == -0.195000000000000);
        assertTrue(x[1] == -0.130000000000000);
        assertTrue(x[6] ==  0.195000000000000);
    }

    @Test
    public void testGenerateCoordSub20(){
        double[] x = Gmatrix.generateCoordinates(7,0.065,20);
        assertTrue(x[0]  == -0.225875000000000);
        assertTrue(x[50] == -0.063375000000000);
        assertTrue(x[100]==  0.099125000000000);
    }

    @Test
    public void testGenerateGfunctionFloat(){
        double[] x_ccd = Gmatrix.generateCoordinates(7,0.065,1);
        //ArrayList<Point2D> ccd = psf.generateListOfPoints(psf.invertArray(x_ccd));
        double[] x_invert = Utils.invertArray(x_ccd);
        ArrayList<Point2D> ccd = Gmatrix.generateListOfPoints(x_invert);
        double[] x_image = Gmatrix.generateCoordinates(7,0.065,20);
        ArrayList<Point2D> image = Gmatrix.generateListOfPoints(x_image);
        float[][] g = Gmatrix.GenerateGMatrixFloat(510,1.49,ccd,image);

        double delta = 1e-06;

        assertEquals(2.275288439569309e-04,g[0][0],delta);
        assertEquals(0.005639107100326,g[33][7],delta);
        assertEquals(2.488481309434887e-04, g[44][17037], delta);
    }

    @Test
    public void testGenerateGfunction510_149_100_6500_20(){
        double imagePixelSize = Gmatrix.calculateImagePixelSize(6500,100);
        int n_w = Gmatrix.calculateWindowSize(510, 1.49, imagePixelSize );
        INDArray gmatrix = Gmatrix.GenerateGMatrix(510, 1.49, n_w,imagePixelSize,20);
        double delta = 1e-06;

        assertEquals(2.275288439569309e-04,gmatrix.getFloat(0,0),delta * 2.275288439569309e-04);
        assertEquals(0.005639107100326,gmatrix.getFloat(33, 7),delta * 0.005639107100326);
        assertEquals(2.488481309434887e-04, gmatrix.getFloat(44, 17037), delta * 2.488481309434887e-04);

    }

    @Test
    public void testGenerateGfunction583_142_1_80_20(){
        double imagePixelSize = Gmatrix.calculateImagePixelSize(80,1);
        int n_w = Gmatrix.calculateWindowSize(583, 1.42, imagePixelSize );
        INDArray gmatrix = Gmatrix.GenerateGMatrix(583, 1.42, n_w,imagePixelSize,20);
        double delta = 1e-06;
        assertEquals(3.347000951010260e-04, gmatrix.getFloat(0,0), delta* 3.347000951010260e-04);
        assertEquals(0.003939536474269, gmatrix.getFloat(33, 7),delta * 0.003939536474269);
        assertEquals(5.961340606405262e-05, gmatrix.getFloat(44, 17037), delta * 5.961340606405262e-05);
    }


}
