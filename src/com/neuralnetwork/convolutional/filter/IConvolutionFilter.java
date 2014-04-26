package com.neuralnetwork.convolutional.filter;

import org.ejml.data.DenseMatrix64F;

public interface IConvolutionFilter
{
    void convolve(DenseMatrix64F input, DenseMatrix64F output);

    IConvolutionFilter setKernel(DenseMatrix64F kernel);
}
