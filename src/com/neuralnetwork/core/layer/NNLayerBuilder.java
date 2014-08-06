package com.neuralnetwork.core.layer;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;

public class NNLayerBuilder {

    Neuron[] aNeurons;

    public NNLayerBuilder setNeurons(Neuron... neurons)
    {
        this.aNeurons = neurons;

        return this;
    }

    public NNLayer build() throws Exception
    {
        validate();

        return new NNLayer(this);
    }

    private void validate() throws Exception
    {
        validateActivationFunction();
        validateNeurons();
    }

    private void validateActivationFunction() throws Exception
    {
        IActivationFunction.IDifferentiableFunction phi = aNeurons[0].phi();

        for(Neuron neuron:aNeurons)
        {
            if (!neuron.phi().equals(phi))
                throw new IllegalArgumentException("Neurons must all use the same activation function");

            phi = neuron.phi();
        }
    }

    private void validateNeurons()
    {
        int size = aNeurons[0].getNumberOfWeights();

        for(Neuron neuron:aNeurons)
        {
            if (size != neuron.getNumberOfWeights())
                throw new IllegalArgumentException("The neurons must form a fully-connected network");

            size = neuron.getNumberOfWeights();
        }
    }
}
