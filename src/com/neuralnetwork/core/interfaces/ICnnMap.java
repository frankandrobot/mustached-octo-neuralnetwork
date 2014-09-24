package com.neuralnetwork.core.interfaces;

import org.ejml.data.DenseMatrix64F;

public interface ICnnMap extends IBasicLayer
{
    DenseMatrix64F generateOutput(DenseMatrix64F... input);

    DenseMatrix64F generateInducedLocalField(DenseMatrix64F... input);

    int getNumberOfInputs();

    DenseMatrix64F getWeightMatrix();
}