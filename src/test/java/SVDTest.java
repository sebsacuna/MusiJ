import linalg.SVD;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

public class SVDTest {

    @Test
    public void test2x2(){
        SVD svd = new SVD(2,2);

        INDArray sample = Nd4j.create(new float[][]{{1,2},{3,4}},'f');

        svd.ComputeSVD(sample);

        assertEquals(5.464985704219043, svd.getS().getFloat(0), 1e-6);
        assertEquals(0.365966190626258, svd.getS().getFloat(1), 1e-6);

        assertEquals(-0.404553584833757, svd.getU().getFloat(0,0), 1e-6);
        assertEquals(-0.914514295677305, svd.getU().getFloat(0,1), 1e-6);
    }

    @Test
    public void test5x2(){
        SVD svd = new SVD(5,2);

        INDArray sample = Nd4j.create(new float[][]{{1,2},{3,4},{5,6},{7,8},{9,10}});

        svd.ComputeSVD(sample);

        assertEquals(19.608156890627935, svd.getS().getFloat(0), 1e-6);
        assertEquals(0.721237375986647, svd.getS().getFloat(1), 1e-6);

        assertEquals(-0.110495838427047, svd.getU().getFloat(0,0), 1e-6);
        assertEquals(-0.766675074389605, svd.getU().getFloat(0,1), 1e-6);
        assertEquals(0.242874112755323, svd.getU().getFloat(4,4), 1e-6);
    }
}
