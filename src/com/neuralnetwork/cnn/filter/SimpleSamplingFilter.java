package com.neuralnetwork.cnn.filter;


import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class SimpleSamplingFilter implements IConvolutionFilter
{
    protected DenseMatrix64F kernel;
    protected DenseMatrix64F temp;

    @Override
    public IConvolutionFilter setKernel(DenseMatrix64F kernel)
    {
        if (kernel.numCols != kernel.numRows)
            throw new IllegalArgumentException("Kernel must be square");

        this.kernel = kernel;
        this.temp = new DenseMatrix64F(kernel.numRows, kernel.numCols);
        return this;
    }

    /**
     * In order for this to work,
     * output.width = input.width / kernel.width
     * output.length = input.length / kernel.length
     *
     * @param input
     * @param output
     */
    @Override
    public void convolve(DenseMatrix64F input, DenseMatrix64F output)
    {
        int kernelDim = kernel.numRows; //this works because the kernel is square man
        int inputDim = input.numRows;   //and so must the input
        int len = inputDim / kernelDim;

        for(int col=0; col < len; ++col)
            for (int row = 0; row < len; ++row)
            {
                int icol = col * kernelDim;
                int irow = row * kernelDim;
                CommonOps.extract(input,
                        irow, irow+kernelDim,
                        icol, icol+kernelDim,
                        temp,
                        0,
                        0);
                CommonOps.elementMult(temp, kernel);
                output.unsafe_set(row,col,CommonOps.elementSum(temp));
            }
    }
}
