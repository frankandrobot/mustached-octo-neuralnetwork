package com.neuralnetwork.cnn.filter;

import org.ejml.data.DenseMatrix64F;
import org.junit.Assert;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;

public class SimpleConvolutionFilterTest {

    @Test
    public void testFilter1() throws Exception
    {
        DenseMatrix64F input = createMatrix(5);
        DenseMatrix64F kernel = createMatrix(2);
        DenseMatrix64F output = createMatrix(5-2+1);

        IFilter filter = new SimpleConvolutionFilter();
        filter.setKernel(kernel);
        filter.convolve(input, output);

        double o00 = input.get(0,0)*kernel.get(0,0) + input.get(0,1)*kernel.get(0,1)
                + input.get(1,0)*kernel.get(1,0) + input.get(1,1)*kernel.get(1,1);
        Assert.assertThat(o00, is(output.get(0,0)));

        double o01 = input.get(0,1)*kernel.get(0,0) + input.get(0,2)*kernel.get(0,1)
                + input.get(1,1)*kernel.get(1,0) + input.get(1,2)*kernel.get(1,1);
        Assert.assertThat(o01, is(output.get(0,1)));

        double o10 = input.get(1,0)*kernel.get(0,0) + input.get(1,1)*kernel.get(0,1)
                + input.get(2,0)*kernel.get(1,0) + input.get(2,1)*kernel.get(1,1);
        Assert.assertThat(o10, is(output.get(1,0)));

        double o11 = input.get(1,1)*kernel.get(0,0) + input.get(1,2)*kernel.get(0,1)
                + input.get(2,1)*kernel.get(1,0) + input.get(2,2)*kernel.get(1,1);
        Assert.assertThat(o11, is(output.get(1,1)));
    }



    private DenseMatrix64F createMatrix(int size)
    {
        double[] input = new double[size*size];
        for(int i=0; i<input.length; i++)
            input[i] = i;

        return new DenseMatrix64F(size,size,true,input);
    }
}
