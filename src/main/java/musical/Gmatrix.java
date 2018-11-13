package musical;

import linalg.Point2D;
import linalg.Utils;
import org.apache.commons.math3.special.BesselJ;
import org.apache.commons.math3.special.Gamma;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;

/**
 * A class that allows creation of G, or PSF matrix
 *
 * @author Sebastian Acuña
 */

public class Gmatrix {

    /**
     * Generates an array representing the coordinates in metric system of the pixels
     * in a grid of N_w. The pixels can be divided into more blocks depending of subpixels
     *
     * @param N_w is the size in pixels
     * @param imagePixSize size of each pixel in metric system
     * @param subpixels subdivision for each pixel
     * @return an array with all the coordinate form negative to positive, centered in 0.
     */

    public static double[] generateCoordinates(int N_w, double imagePixSize, int subpixels){
        BigDecimal step = BigDecimal.valueOf(imagePixSize).divide(BigDecimal.valueOf(subpixels), 8, RoundingMode.HALF_UP);
        BigDecimal step_half = step.divide(BigDecimal.valueOf(2));
        BigDecimal side_length_half = BigDecimal.valueOf(N_w).multiply(BigDecimal.valueOf(imagePixSize)).divide(BigDecimal.valueOf(2),8, RoundingMode.HALF_UP);
        BigDecimal shift = step_half.subtract(side_length_half);
        int side_n_pixels = N_w*subpixels;
        double[] x = new double[side_n_pixels];
        for(int i = 0;i<side_n_pixels;i++){
            //x[i] = i*step + step/2 - side_length/2;
            x[i] = step.multiply(BigDecimal.valueOf(i)).add(shift).doubleValue();
        }
        return x;
    }

    /**
     * Generates the G matrix, which represents the effect of particles emiting inside
     * a subgrid. Read the paper to understand.
     *
     * @param lam is the emission wavelength in nm (Eg. 510)
     * @param na is numerical aperture (Eg. 1.49)
     * @param ccd coordinate of the pixels in the physical ccd
     * @param image coordinates of the pixels in the image
     * @return a 2d array of floats containing the G matrix.
     */

    public static float[][] GenerateGMatrixFloat(double lam, double na, ArrayList<Point2D> ccd, ArrayList<Point2D> image){

        BesselJ j1 = new BesselJ(1);
        double k = 2*Math.PI/(lam/1000);

        int pixels_ccd   = ccd.size();
        int pixels_image = image.size();

        float[][] g = new float[pixels_ccd][pixels_image];

        double aux = 0;

        double singularity = 0.5/ Gamma.gamma(2);
        singularity = singularity*singularity;

        for(int i = 0;i<pixels_ccd;i++){
            for(int j = 0;j<pixels_image;j++){
                double rho = ccd.get(i).distance(image.get(j));
                double rho_=na*rho;

                if(rho_!=0){
                    aux = j1.value(k*rho_)/(k*rho_);
                    g[i][j] = (float)(aux*aux);
                }
                else{
                    g[i][j] = (float)singularity;
                }
            }
        }
        return g;
    }

    /**
     * Generates a list of 2d points where each one is a 2d coordinate of a pixel in metric
     * system of a grid. Assumes we are using a square grid.
     *
     * @param x linear array representing the coordinates
     * @return generates a list of all the points of a grid
     */

    public static ArrayList<Point2D> generateListOfPoints(double[] x){
        ArrayList<Point2D> list = new ArrayList<>();
        for(int i = 0;i<x.length;i++){
            for(int j = 0;j<x.length;j++){
                list.add(new Point2D(x[i],x[j]));
            }
        }
        return list;
    }

    /**
     * Generate the G matrix, and returns it as a Nd4j 2d matrix.
     *
     * @param n_w size of the grid
     * @param imagePixSize is the size of each pixel in the image
     * @param subpixels number of subpixels to divide
     * @return a 2d Nd4j matrix
     */

    public static INDArray GenerateGMatrix(double lam, double na, int n_w, double imagePixSize, int subpixels){
        double[] x_ccd = generateCoordinates(n_w,imagePixSize,1);
        //ArrayList<Point2D> ccd = psf.generateListOfPoints(psf.invertArray(x_ccd));
        double[] x_invert = Utils.invertArray(x_ccd);
        ArrayList<Point2D> ccd = generateListOfPoints(x_invert);
        double[] x_image = generateCoordinates(n_w,imagePixSize,subpixels);
        ArrayList<Point2D> image = generateListOfPoints(x_image);

        float[][] g = GenerateGMatrixFloat(lam,na,ccd,image);

        return Nd4j.create(g);
    }

    /**
     * Computes the size of the window using the optical parameters
     *
     * @param lam is the emission wavelegnth (lambda) in [nm] (Eg. 510)
     * @param na is the numerical aperture
     * @param imagePixSize is the size of each pixel in the image
     * @return the size of the grid. Will always be an odd number.
     */

    public static int calculateWindowSize(double lam, double na, double imagePixSize){
        double lam_u = lam/1000;            // [um]
        // Diameter of the airy disk (the main disk) in pixel units.
        // We use this number as the size of the window, and we choose it to be even.
        return((int)(1+2*Math.floor(0.61*lam_u/(na*imagePixSize))));
    }

    /**
     * Compute the pixel size in the image, using the pixel size in the CCD and
     * the magnification of the optical system. Re
     *
     * @param pixsize is the size of the pixel in the CCD in nm
     * @param mag is the magnification of the optical system
     * @return the size of a pixel in the image in um
     */

    public static double calculateImagePixelSize(double pixsize, double mag){
        return pixsize/mag/1000.0;
    }


}
