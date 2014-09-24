package com.neuralnetwork.core.interfaces;

import org.ejml.data.DenseMatrix64F;

public interface IMatrixNeuralLayer extends IBasicLayer
{
    public DenseMatrix64F generateOutput(DenseMatrix64F[] input);

    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F[] input);

    public int getNumberOfInputs();

    public DenseMatrix64F getWeightMatrix();

    public IActivationFunction.IDifferentiableFunction getImpulseFunction();
}