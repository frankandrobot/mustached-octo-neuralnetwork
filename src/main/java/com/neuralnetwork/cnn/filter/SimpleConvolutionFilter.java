package com.neuralnetwork.cnn.filter;


import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class SimpleConvolutionFilter implements IConvolutionFilter
{
    protected DenseMatrix64F kernel;
    protected DenseMatrix64F temp;

    @Override
    public IFilter setKernel(DenseMatrix64F kernel)
    {
        if (kernel.numCols != kernel.numRows)
            throw new IllegalArgumentException("Kernel must be square");

        this.kernel = kernel;
        this.temp = new DenseMatrix64F(kernel.numRows, kernel.numCols);

        return this;
    }

    /**
     * In order for this to work,
     * output.width = input.width - kernel.width + 1
     * output.length = input.length - kernel.length + 1
     *
     * @param input
     * @param output
     */
    @Override
    public void convolve(DenseMatrix64F input, DenseMatrix64F output)
    {
        int kernelDim = kernel.numRows;

        for(int col=0; col <= input.numCols - kernelDim; ++col)
            for (int row = 0; row <= input.numRows - kernelDim; ++row)
            {
                //copy over a kernel-sized copy of the input into $tmp starting at (row,col)
                CommonOps.extract(input,
                        row, row+kernel.numRows,
                        col, col+kernel.numCols,
                        temp,
                        0,
                        0);

                //multiply by the kernel
                CommonOps.elementMult(temp, kernel);

                //save result in output
                output.unsafe_set(row,col,CommonOps.elementSum(temp));
            }
    }
}
