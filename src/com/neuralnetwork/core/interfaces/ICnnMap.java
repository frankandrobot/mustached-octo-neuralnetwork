package com.neuralnetwork.core.interfaces;

import org.ejml.data.DenseMatrix64F;

import java.util.List;

public interface ICnnMap extends IBasicLayer
{
    DenseMatrix64F generateOutput(DenseMatrix64F... input);

    DenseMatrix64F generateInducedLocalField(DenseMatrix64F... input);

    int getNumberOfInputs();

    DenseMatrix64F getWeightMatrix();

    void validateInputs(List<ICnnMap> inputMaps);
}