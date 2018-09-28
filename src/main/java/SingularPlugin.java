import Exceptions.WrongShape;
import ij.gui.Plot;
import images.ImageReader;
import io.scif.services.DatasetIOService;
import linalg.Linalg;
import linalg.Utils;
import musical.GaussianMask;
import musical.Gmatrix;
import musical.Singular;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.scijava.ItemIO;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.awt.*;
import java.util.List;

/**
 * Plugin that allows computation of singlar values of a single channel stack image
 *
 * @author Sebastian Acuña
 */

@Plugin(type = Command.class,
        headless = true,
        menuPath = "Plugins>Musical>Functions>Singular Values",
        label = "JMusical v0.9 : Singular Values - 2018")
public class SingularPlugin <T extends RealType<T> & NativeType< T >> implements Command {

    /*
     * Services
     */

    @Parameter
    private DatasetIOService datasetIOService;

    @Parameter
    private DatasetService datasetService;

    @Parameter
    private LogService logService;

    @Parameter
    private StatusService ss;

    /*
     * Input image
     */

    @Parameter(type = ItemIO.INPUT)
    private Dataset inputImageDataset;

    /*
     * Optical parameters
     */

    @Parameter(label = "Emission [nm]", description = "Wavelength of emission")
    private double em = 510;

    @Parameter(label = "Numerical Aperture", description = "Numerical Aperture of the optical system")
    private double na = 1.49;

    @Parameter(label = "Magnification", description = "Magnification of the optical system")
    private double mag = 100;

    @Parameter(label = "Pixel size [nm]", description = "Size of the pixel in the camera")
    private double pixsize = 6500;

    @Override
    public void run(){
        ImgPlus<T> inputImgPlus = (ImgPlus<T>)inputImageDataset.getImgPlus();
        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            ss.warn(wrongShape.getMessage());
            return;
        }

        double imagePixelSize = Gmatrix.calculateImagePixelSize(pixsize, mag);
        int n_w = Gmatrix.calculateWindowSize(em,na, imagePixelSize);
        if (n_w < 7){
            n_w = 7;
        }

        logService.info("Computing singular values on " + inputImageDataset.getName());

        logService.info("Emission: " + em + " nm");
        logService.info("Numerical aperture: " + na);
        logService.info("Magnification: " + mag);
        logService.info("Pixel size: " + pixsize + " nm");

        ss.showStatus("Using n_w:  " + n_w);
        logService.info("Using n_w:  " + n_w);
        int pad = GaussianMask.calculatePadding(n_w);
        INDArray inputMatrixPadded = Linalg.pad(inputMatrix, pad);
        INDArray mask = GaussianMask.getMask(n_w);
        ss.showStatus("Computing singular values");
        List<INDArray> svlist = Singular.SingularValues(inputMatrixPadded, mask);
        ss.showStatus("Done");
        logService.info("Singular values done");
        String fname = inputImageDataset.getName();
        plotSingularValues(svlist, fname);
    }

    public void plotSingularValues(List<INDArray> sv, String title)
    {
        int total_color = sv.size();

        Plot plt = new Plot("Singular Values: " + title,"Number","Log10 Value");

        //plt.setAxisYLog(true);
        float maxIndex = (float)sv.get(0).length();
        float maxValue = 0;
        plt.setLineWidth(1);
        int color_index = 0;
        for(INDArray dm: sv){
            float[] y = new float[(int)dm.length()];
            for(int i = 0;i<dm.length();i++){
                y[i] = dm.getFloat(i);
                if(y[i]>maxValue){
                    maxValue = y[i];
                }
            }
            float[] x = new float[y.length];
            for(int i = 0;i<x.length;i++){
                x[i] = i;
            }
            y = Utils.log10array(y);
            plt.setColor(Color.getHSBColor((float)(color_index)/(total_color),1,1));
            plt.addPoints(x,y,Plot.LINE);
            color_index++;
        }
        plt.setLimits(0,maxIndex,-8,maxValue);
        plt.show();
    }

    public static void main(String[] args) {
        final ImageJ ij = new ImageJ();
        ij.launch(args);
        ij.command().run(SingularPlugin.class, true);
    }
}
