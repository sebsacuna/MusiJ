package linalg;

public interface IMatrix {

    // Setters
    void SetElement(int dim1, int dim2, float value);
    void SetElement(int dim1, int dim2, int dim3, float value);

    // Getters
    float GetElement(int dim1, int dim2);
    float GetElement(int dim1, int dim2, int dim3);


    /**
     * Normalize the matrix by the maximum value
     */
    void NormMax();

    /**
     * Shape of the matrix
     *
     * @return
     */
    long[] Shape();

    // FUNCTIONS


}
