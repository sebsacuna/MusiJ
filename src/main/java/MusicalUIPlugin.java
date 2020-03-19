import io.scif.services.DatasetIOService;
import musical.Gmatrix;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.command.InteractiveImageCommand;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import org.scijava.widget.Button;

import java.io.File;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

/**
 * A GUI for applying SingularPlugin and MusicalPlugin
 *
 * @author Sebastian Acuña
 */

@Plugin(type = Command.class,
        initializer = "updateParameters",
        label = "MusiJ v0.94 - 2019",
        headless = true, menuPath = "Plugins>MUSICAL>Musical UI")
public class MusicalUIPlugin extends InteractiveImageCommand {

    private static final String TITLE_LABEL =   "JMusical v0.94";
    private static final String CREATOR =       "Krishna Agarwal [uthkrishth@gmail.com]";
    private static final String DEVELOPER =     "Sebastian Acuna [sebacuma@gmail.com]\n";
    private static final String OPTI_LABEL =    "----- Optical parameters ---";
    private static final String MUSI_LABEL =    "----- Musical parameters ---";

    // TOP HEADER

    @Parameter(visibility = ItemVisibility.MESSAGE, label = " ", persist = false)
    private final String title = TITLE_LABEL;

    @Parameter(visibility = ItemVisibility.MESSAGE, label = "Created by: ", persist = false)
    private final String creator = CREATOR;

    @Parameter(visibility = ItemVisibility.MESSAGE, label = "Implemented by:  ", persist = false)
    private final String developer = DEVELOPER;

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

    @Parameter(label = "To be applied to: ", visibility = ItemVisibility.MESSAGE, persist = false)
    private String image_name = "";

    /*
     * Parameters
     */

    @Parameter(visibility = ItemVisibility.MESSAGE, label = " ")
    private final String optical_param = OPTI_LABEL;

    @Parameter(label = "Emission [nm]", description = "Wavelength of emission in nanometers")
    private double em = 510;

    @Parameter(label = "Numerical Aperture", description = "Numerical Aperture of the optical system")
    private double na = 1.49;

    @Parameter(label = "Magnification", description = "Optical magnification of the optical system")
    private double mag = 100;

    @Parameter(label = "Pixel size [nm]", description = "Physical size of the camera's pixel in nanometers")
    private double pixsize = 6500;

    @Parameter(label = "N_w", visibility = ItemVisibility.MESSAGE, persist = false,
            description = "The size is computed used the optical parameters. The minimum is 7")
    private String n_w_string = "";

    private int n_w = 7;

    @Parameter(label = "Plot singular values", callback = "SingularValuesCallback")
    private Button plot_s_values;

    /*
     * MUSICAL PARAMETERS
     */


    @Parameter(visibility = ItemVisibility.MESSAGE, label = " ")
    private String music_param = MUSI_LABEL;

    @Parameter(label = "Threshold", description = "Defines the threshold for the singular values (log10)")
    private double threshold = -1;

    @Parameter(label = "Alpha")
    private double alpha = 4;

    @Parameter(callback = "updateParameters", label = "Subpixels per pixel", description = "Number of subpixels to be calcUlated along one side of each pixel")
    private int subpixels = 20;

    @Parameter(label = "Multithreading")
    private boolean multithreading = false;

    @Parameter(label = "Threads (if Multithreading)", min = "1")
    private int threads = 1;

    @Parameter(label = "Autosave")
    private boolean save = false;

    @Parameter(label = "Output", style = DIRECTORY_STYLE, required = false)
    private File output;

    @Parameter(label = "Generate Image", callback = "MusicalImageCallback")
    private Button start_musical;


    /*
     * Callbacks
     */


    /*
     * The actual value of n_w to use is calculated again in the function. This should be fixed in the
     * future.
     */

    public void updateParameters(){
        double imgPixelSize = Gmatrix.calculateImagePixelSize(pixsize, mag);
        n_w = Gmatrix.calculateWindowSize(em, na, imgPixelSize );

        if (n_w < 7){
            n_w_string = "7 (computed value is " + n_w+ ")";
            n_w = 7;
        }
        else {
            n_w_string = String.valueOf(n_w);
        }
    }

    public void putName(){
        image_name = inputImageDataset.getName();
    }

    public void SingularValuesCallback(){
        ss.showStatus("Running Singular Values...");
        cs.run(SingularPlugin.class, true,
                "inputImageDataset", inputImageDataset, "em", em, "na", na, "mag", mag, "pixsize", pixsize );
        ss.showStatus("Done");
    }

    public void MusicalImageCallback(){
        ss.showStatus("Running MUSICAL Image...");
        cs.run(MusicalPlugin.class, true,
                "inputImageDataset", inputImageDataset, "em", em, "na", na, "mag", mag, "pixsize", pixsize,
                "subpixels", subpixels, "threshold", threshold, "alpha", alpha, "multithreading", multithreading,
                "threads", threads, "save", save, "output", output);

    }

    public static void main(String[] args) {
        final ImageJ ij = new ImageJ();
        ij.launch(args);

        ij.command().run(MusicalUIPlugin.class, true);
    }
}
