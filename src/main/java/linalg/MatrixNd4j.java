package linalg;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MatrixNd4j implements IMatrix {

    private INDArray data;

    public MatrixNd4j(int dim1, int dim2, int dim3)
    {
        //data = Nd4j.zeros(new long[]{(int)dim1, (int)dim2, (int)dim3});
        data = Nd4j.zeros(new int[]{(int)dim1, (int)dim2, (int)dim3});
    }

    public MatrixNd4j(int dim1, int dim2)
    {
        data = Nd4j.zeros((int)dim1, (int)dim2);
    }

    public void setData(INDArray data){
        this.data = data;
    }

    @Override
    public void SetElement(int dim1, int dim2, float value) {
        data.putScalar(new int[]{dim1, dim2}, value);
    }

    @Override
    public void SetElement(int dim1, int dim2, int dim3, float value) {
        data.putScalar(new int[]{dim1, dim2, dim3}, value);
    }

    @Override
    public float GetElement(int dim1, int dim2) {
        return data.getFloat(new int[]{dim1, dim2});
    }

    @Override
    public float GetElement(int dim1, int dim2, int dim3) {
        return data.getFloat(new int[]{dim1, dim2, dim3});
    }

    @Override
    public void NormMax(){
        data.divi(data.maxNumber().floatValue());
    }

    @Override
    public long[] Shape(){
        return data.shape();
    }



}
