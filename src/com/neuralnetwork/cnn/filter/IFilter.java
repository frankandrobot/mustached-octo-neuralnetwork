package com.neuralnetwork.cnn.filter;

import org.ejml.data.DenseMatrix64F;

public interface IFilter
{
    void convolve(DenseMatrix64F input, DenseMatrix64F output);

    IFilter setKernel(DenseMatrix64F kernel);
}
