package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.helpers.Dimension;
import org.ejml.data.DenseMatrix64F;

public interface IMatrixNeuralLayer
{
    public DenseMatrix64F generateY(DenseMatrix64F[] input);

    public DenseMatrix64F generateOutput(DenseMatrix64F[] input);

    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F[] input);

    public Dimension getInputDim();

    public Dimension getOutputDim();

    public int getNumberOfNeurons();

    public DenseMatrix64F getWeightMatrix();

    public IActivationFunction.IDifferentiableFunction getImpulseFunction();
}