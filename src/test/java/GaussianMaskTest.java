import musical.GaussianMask;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static junit.framework.TestCase.assertEquals;

public class GaussianMaskTest {

    @Test
    public void testMask7x7(){
        INDArray mask = GaussianMask.getMask(7);

        assertEquals(0.049787068367864, mask.getFloat(0,0), 1e-6);
        assertEquals(0.188875602837562, mask.getFloat(0,2),1e-6);
        assertEquals(1.0, mask.getFloat(3,3),1e-6);
        assertEquals(0.049787068367864, mask.getFloat(6,6),1e-6);
    }

}
