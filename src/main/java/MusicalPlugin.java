import Exceptions.WrongShape;
import images.ImageReader;
import io.scif.services.DatasetIOService;
import musical.*;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.scijava.ItemIO;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.display.event.input.KyEvent;
import org.scijava.event.EventHandler;
import org.scijava.log.LogService;
import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

/**
 * Plugin implementing MUSICAL for single channel stack image
 *
 * @author Sebastian Acuña
 */

@Plugin(type = Command.class,
        headless = true,
        menuPath = "Plugins>MUSICAL>Functions>MUSICAL Image",
        label = "MusiJ v0.94 : MUSICAL Image - 2019")
public class MusicalPlugin  <T extends RealType<T> & NativeType< T >> implements Command {

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
    private UIService uiService;

    @Parameter
    private StatusService ss;

    @Parameter
    private CommandService cs;

    @Parameter
    private ModuleService ms;

    /*
     * Input / Output image
     */

    @Parameter(type = ItemIO.INPUT, validater = "putName")
    private Dataset inputImageDataset;

    @Parameter(type = ItemIO.OUTPUT)
    private Dataset outputImageDataset;

    /*
     * Parameters
     */

    @Parameter(label = "Emission [nm]", description = "Wavelength of emission in nanometers")
    private double em = 510;

    @Parameter(label = "Numerical Aperture", description = "Numerical Aperture of the optical system")
    private double na = 1.49;

    @Parameter(label = "Magnification", description = "Optical magnification of the optical system")
    private double mag = 100;

    @Parameter(label = "Pixel size [nm]", description = "Physical size of the camera's pixel in nanometers")
    private double pixsize = 6500;

    /*
     * MUSICAL PARAMETERS
     */

    @Parameter(label = "Threshold", description = "Defines the threshold for the singular values (log10)")
    private double threshold = -1;

    @Parameter(label = "Alpha")
    private double alpha = 4;

    @Parameter(label = "Subpixels per pixel", description = "Number of subpixels to be created along one side of each pixel")
    private int subpixels = 20;

    @Parameter(label = "Multithreading")
    private boolean multithreading = false;

    @Parameter(label = "Threads (if multithreading)", min = "1")
    private int threads = 1;

    @Parameter(label = "Autosave")
    private boolean save = false;

    @Parameter(label = "Output", style = DIRECTORY_STYLE, required = false)
    private File output;


    private KeyEvent ke;

    @Override
    public void run(){

        //Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
        //nd4jBlas.setMaxThreads(1);
        //NativeOpsHolder instance = NativeOpsHolder.getInstance();
        //NativeOps deviceNativeOps = instance.getDeviceNativeOps();
        //deviceNativeOps.setOmpNumThreads(1);

        //System.out.println("Blas          N: " + nd4jBlas.getMaxThreads());
        //System.out.println("Native        N: " + deviceNativeOps.ompGetNumThreads());
        //System.out.println("Native (max)  N: " + deviceNativeOps.ompGetMaxThreads());

        //ke = new KeyEvent();
        //this.context().inject(ke);

        if(save && (output == null || !output.exists())) {
            ss.warn("Invalid output directory");
            return;
        }

        ImgPlus<T> inputImgPlus = (ImgPlus<T>)inputImageDataset.getImgPlus();

        INDArray inputMatrix = null;
        try {
            inputMatrix = ImageReader.ImgToMatrix(inputImgPlus, true);
        } catch (WrongShape wrongShape) {
            ss.warn(wrongShape.getMessage());
            return;
        }

        double imagePixelSize = Gmatrix.calculateImagePixelSize(pixsize,mag);
        int n_w = Gmatrix.calculateWindowSize(em,na, imagePixelSize);
        if (n_w < 7){
            n_w = 7;
        }
        logService.info("Computing MUSICAL image on " + inputImageDataset.getName());
        logService.info("Emission: " + em + "nm");
        logService.info("Numerical aperture: " + na);
        logService.info("Magnification: " + mag);
        logService.info("Pixel size: " + pixsize + "nm");
        logService.info("Threshold: " + threshold);
        logService.info("Alpha " + alpha);
        logService.info("Subpixels: " + subpixels);
        logService.info("Multithreading: " + multithreading);
        if(multithreading){
            logService.info("Threads: " + threads);
        }

        logService.info("Using n_w:  " + n_w);
        int pad = GaussianMask.calculatePadding(n_w);
        INDArray mask = GaussianMask.getMask(n_w);
        //INDArray inputMatrixPadded = Linalg.pad(inputMatrix, pad);
        logService.info("Computing g");
        ss.showStatus("Computing g");
        INDArray gmatrix = Gmatrix.GenerateGMatrix(em, na, n_w, imagePixelSize, subpixels);
        ss.showStatus("Computing MUSICAL image");
        logService.info("Computing MUSICAL image");

        long[] input_shape = inputMatrix.shape();
        Status s = new Status(0, (int)(input_shape[0] * input_shape[1]));

        Thread t1 = new Thread(() -> {
            while (!s.done) {
                ss.showStatus(s.progress, s.total, "Working on MUSICAL image...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    System.out.println("There was an error with the progress bar");
                }
            }
            ss.showStatus(100, 100, "MUSICAL image done");
        });
        t1.start();

        long t = System.currentTimeMillis();
        INDArray musical_result;

        if(!multithreading) {
            musical_result = Musical2.WrapperGetMusical(inputMatrix, mask,gmatrix, subpixels,(float)threshold,(float)alpha, s);
            //musical_result = Musical.WrapperGetMusical(inputMatrix, mask,gmatrix, subpixels,(float)threshold,(float)alpha);
        }
        else {
            musical_result = Musical2.WrapperGetMusical(inputMatrix, mask, gmatrix, subpixels, (float) threshold, (float) alpha, threads, s);
        }
        //INDArray outputMatrixPadded = Linalg.pad(inputMatrix, pad * subpixels);
        long tfinal = (System.currentTimeMillis() - t)/1000;
        s.done = true;
        //System.out.println("Total time: " + tfinal + " seconds.");
        ss.showStatus("MUSICAL done: " + tfinal + " seconds");
        logService.info("MUSICAL done: " + tfinal + " seconds");

        String name = inputImageDataset.getName();
        String[] fname = name.split("\\.");
        Date currentTime = Calendar.getInstance().getTime();
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(currentTime);
        String title = fname[0] + "_musical_100t" + (int)(100*threshold) + "_" + timeStamp;
        String titlext = title + ".tiff";

        outputImageDataset = datasetService.create(ImageReader.MatrixToImg(musical_result, new UnsignedShortType(),title));

        if(save){
            String inputPath = inputImageDataset.getSource();
            File outputFile = new File(output, titlext);

            try {
                datasetIOService.save(outputImageDataset, outputFile.getPath());
            } catch (IOException e) {
                e.printStackTrace();
                ss.warn("Image could not be saved. Do it manually");
            }
            File logFile = new File(output, title + ".txt");

            try {
                PrintWriter writer = new PrintWriter(logFile);
                String log = printInfo(inputPath,outputFile.getPath(),currentTime,
                        em, na, mag, pixsize, subpixels, n_w, threshold, alpha, multithreading, threads);
                writer.print(log);
                writer.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
                ss.warn("Could not save logFile");
            }
        }
    }

    public class KeyEvent{
        @Parameter
        LogService logService;

        @EventHandler
        public void onEvent(final KyEvent ev){
            logService.info(ev);
        }
    }

    public String printInfo(String originalFile,
                            String resultFile,
                            Date timeStamp,
                            double emission,
                            double na,
                            double mag,
                            double pixelsize,
                            double subpixels,
                            int n_w,
                            double threshold,
                            double alpha,
                            boolean multithreading,
                            int threads
    ){

        StringBuilder sb = new StringBuilder();
        sb.append("Musical 0.94:\r\n\r\n");

        sb.append("Time:                " + timeStamp.toString()+"\r\n");

        sb.append("Originak file:       " + originalFile + "\r\n");
        sb.append("Result:              " + resultFile + "\r\n");

        sb.append("Emission [nm]:       " + emission + "\r\n");
        sb.append("Numerical aperture:  " + na +"\r\n");
        sb.append("Magnification:       " + mag + "\r\n");
        sb.append("Pixel size [nm]:     " + pixelsize + "\r\n");
        sb.append("Subpixels:           " + subpixels + "\r\n\r\n");
        sb.append("Window size:         " + n_w + "\r\n");
        sb.append("Threshold:           " + threshold + "\r\n");
        sb.append("Alpha:               " + alpha + "\r\n");
        sb.append("Multithreading:      " + multithreading + "\r\n");
        if(multithreading){
            sb.append("Threads:             " + threads + "\r\n");
        }

        return sb.toString();
    }

    public static void main(String[] args) {
        final ImageJ ij = new ImageJ();
        ij.launch(args);
        ij.command().run(MusicalPlugin.class, true);
    }


}