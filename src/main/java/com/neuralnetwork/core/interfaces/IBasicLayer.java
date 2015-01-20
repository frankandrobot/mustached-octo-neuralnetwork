package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.Dimension;

public interface IBasicLayer
{
    Dimension getInputDim();

    Dimension getOutputDim();

    /**
     * For simplicity, each neuron has the same impulse function.
     *
     * @return
     */
    IActivationFunction.IDifferentiableFunction getImpulseFunction();
}
